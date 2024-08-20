# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk==0.28.0 databricks-agents mlflow-skinny mlflow mlflow[gateway] databricks-vectorsearch langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00-helper

# COMMAND ----------

# chatBotModel = "databricks-dbrx-instruct"
chatBotModel = "databricks-meta-llama-3-70b-instruct"
max_tokens = 2000
VECTOR_SEARCH_ENDPOINT_NAME = "one-env-shared-endpoint-8"
vectorSearchIndexName = "pdf_content_embeddings_index"
embeddings_endpoint = "databricks-bge-large-en"
catalog = "hz_rag_poc_test_catalog"
dbName = "hz_rag_poc_test_db"

finalchatBotModelName = "hz_rag_pdf_test_bot"


# COMMAND ----------

spark.sql(f"USE {catalog}.{dbName}")

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC ## 1 - Prepare our chatbot model with RAG using DBRX

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Configuring our Chain parameters

# COMMAND ----------

# For this first basic demo, we'll keep the configuration as a minimum. In real app, you can make all your RAG as a param (such as your prompt template to easily test different prompts!)
chain_config = {
    "llm_model_serving_endpoint_name": "databricks-dbrx-instruct",  # the foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,  # the endoint we want to use for vector search
    "vector_search_index": f"{catalog}.{dbName}.{vectorSearchIndexName}",
    "embeddings_endpoint": embeddings_endpoint,
    "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""", # LLM Prompt template
}

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1.2 Building our Langchain retriever
# MAGIC
# MAGIC Langchain retriever will be in charge of:
# MAGIC
# MAGIC * Creating the input question (our Managed Vector Search Index will compute the embeddings for us)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with 
# MAGIC
# MAGIC Databricks Langchain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

# We'll register the chain as an MLflow model and inspect the MLflow Trace to understand what is happening inside the chain 

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import mlflow

## Enable MLflow Tracing
# Traces will be logged to the active MLflow Experiment when calling invocation APIs on chains
mlflow.langchain.autolog()

## Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config=chain_config)

## Turn the Vector Search index into a LangChain retriever
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index"),
)
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="content",
    embedding=DatabricksEmbeddings(endpoint=model_config.get("embeddings_endpoint")),
    columns=["id", "content", "url"],
).as_retriever(search_kwargs={"k": 3})

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

# Let's try our retriever chain:
relevant_docs = (vector_search_as_retriever | RunnableLambda(format_context)| StrOutputParser()).invoke('How to start a Databricks cluster?') 


display_txt_as_html(relevant_docs)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### 1.3 Building Databricks Chat Model to query foundation model endpoint

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter

prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", model_config.get("llm_prompt_template")), # Contains the instructions from the configuration
        ("user", "{question}") #user's questions
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

#Let's try our prompt:
answer = (prompt | model | StrOutputParser()).invoke({'question':'How to start a Databricks cluster?', 'context': ''})
display_txt_as_html(answer)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### 1.4 Putting it together in a final chain, supporting the standard Chat Completion format
# MAGIC
# MAGIC We will make sure our chain support the standard Chat Completion API input schema : `{"messages": [{"role": "user", "content": "What is Retrieval-augmented Generation?"}]}`
# MAGIC

# COMMAND ----------

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    } #retrieval chain
    | prompt
    | model
    | StrOutputParser()
)

# COMMAND ----------

# Let's give it a try:
input_example = {"messages": [ {"role": "user", "content": "What is Retrieval-augmented Generation?"}]}
answer = chain.invoke(input_example)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Deploy a RAG Chain to a web-based UI for stakeholder feedback
# MAGIC
# MAGIC Our chain is now ready! 
# MAGIC
# MAGIC Now, we:
# MAGIC 1. Register the Rag Chain application in Unity Catalog
# MAGIC 2. Use Agent Framework to deploy to the review application. This review application is backed by a scalable, production-ready Model Serving endpoint, and is designed to gather stakeholder feedback

# COMMAND ----------

# DBTITLE 1,1. Register the chain in Unity Catalog
# Log the model to MLflow
with mlflow.start_run(run_name=f"{finalchatBotModelName}_run"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), 'chain'),  # Chain code file e.g., /path/to/the/chain.py 
          model_config=chain_config, # Chain configuration 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=input_example,
          example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
      )

model_name = f"{catalog}.{dbName}.{finalchatBotModelName}"

# Register to UC
mlflow.set_registry_uri('databricks-uc')
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_name)

# COMMAND ----------

# DBTITLE 1,2. Deploy the review application
from databricks import agents
# Deploy to enable the Review APP and create an API endpoint
# Note: scaling down to zero will provide unexpected behavior for the chat app. Set it to false for a prod-ready application.
deployment_info = agents.deploy(model_name, model_version=uc_registered_model_info.version, scale_to_zero=True)

instructions_to_reviewer = f"""## Instructions for Testing the Databricks Documentation Assistant chatbot

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement."""

# Add the user-facing instructions to the Review App
agents.set_review_instructions(model_name, instructions_to_reviewer)

wait_for_model_serving_endpoint_to_be_ready(deployment_info.endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - Use the Mosaic AI Agent Evaluation to evaluate your RAG applications
# MAGIC
# MAGIC ### 2.1 Chat with your bot and build your evaluation dataset!
# MAGIC
# MAGIC Our Chat Bot is now live. Normally, you would now give access to internal domain experts and have them test and review the bot. **Your domain experts do NOT need to have Databricks Workspace access** - you can assign permissions to any user in your SSO if you have enabled [SCIM](https://docs.databricks.com/en/admin/users-groups/scim/index.html)
# MAGIC
# MAGIC This is a critical step to build or improve your evaluation dataset: have users ask questions to your bot, and provide the bot with output answer when they don't answer properly.
# MAGIC
# MAGIC Your application is automatically capturing all stakeholder questions and bot responses, including an MLflow trace for each, into Delta Tables in your Lakehouse. On top of that, Databricks makes it easy to track feedback from your end user: if the chatbot doesn't give a good answer and the user gives a thumbdown, their feedback is included in the Delta Tables.
# MAGIC
# MAGIC Your evaluation dataset forms the basis of your development workflow to improve quality: identifying the root causes of quality issues and then objectively measuring the impact of your fixes.
# MAGIC
# MAGIC Once your eval dataset is ready, you'll then be able to leverage it for offline evaluation to measure your new chatbot performance, and also potentially to Fine Tune your model.
# MAGIC <br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/eval-framework.gif?raw=true" width="1000px">
# MAGIC

# COMMAND ----------

print(f"\n\nReview App URL to share with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2.2 Evaluate your bot's quality with Mosaic AI Agent Evaluation specialized LLM judge models
# MAGIC
# MAGIC Our bot is now Live. 
# MAGIC
# MAGIC Evaluation is a key part of deploying a RAG application. Databricks simplify this tasks with specialized LLM models tuned to evaluate your bot's quality/cost/latency, even if ground truth is not available.
# MAGIC
# MAGIC This Agent Evaluation's specialized AI evaluator is integrated into integrated into `mlflow.evaluate(...)`, all you need to do is pass `model_type="databricks-agent"`.
# MAGIC
# MAGIC Mosaic AI Agent Evaluation evaluates:
# MAGIC 1. Answer correctness - requires ground truth
# MAGIC 2. Hallucination / groundness - no ground truth required
# MAGIC 3. Answer relevance - no ground truth required
# MAGIC 4. Retrieval precision - no ground truth required
# MAGIC 5. (Lack of) Toxicity - no ground truth required
# MAGIC
# MAGIC In this example, we'll use an evaluation set that we curated based on our internal experts using the Mosaic AI Agent Evaluation review app interface.  This proper Eval Dataset is saved as a Delta Table.
# MAGIC
# MAGIC To see how to collect the dataset from the Eval App, see the [03-advanced-app/03-Offline-Evaluation]($../03-advanced-app/03-Offline-Evaluation) notebook.

# COMMAND ----------

eval_dataset = spark.table("eval_set_databricks_documentation").limit(10).toPandas()
display(eval_dataset)

# COMMAND ----------

with mlflow.start_run(run_id=logged_chain_info.run_id):
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_dataset, # Your evaluation set
        model=logged_chain_info.model_uri, # previously logged model
        model_type="databricks-agent", # active Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can open your MLFlow Experiment to review the different evaluation, and compare multiple model response to see how different prompts answer: 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-mlflow-eval.png?raw=true" width="1200px">

# COMMAND ----------


