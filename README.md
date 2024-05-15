# Introduction

LLMs are disrupting the way we interact with information, from internal knowledge bases to external, customer-facing documentation or support.

Learn how to create and deploy a real-time Q&A chatbot using Databricks retrieval augmented generation (RAG) and serverless capabilities, leveraging the DBRX Instruct Foundation Model for smart responses.

RAG is a powerful technique where we enrich the LLM prompt with additional context specific to your domain so that the model can provide better answers.

This technique provides excellent results using public models without having to deploy and fine-tune your own LLMs.

In this demo notebook, you will learn how to:

- Prepare clean documents to build your internal knowledge base and specialize your chatbot
- Leverage Databricks Vector Search with our Foundation Model endpoint to create and store document embeddings
- Search similar documents from our knowledge database with Databricks Vector Search
- Deploy a real-time model using RAG and providing augmented context in the prompt
- Leverage the DBRX instruct model through with Databricks Foundation Model endpoint (fully managed)

<br/>

### Alternative option

If you don't have access to a databricks workspace, view those notebooks online: 
<br/>https://notebooks.databricks.com/demos/llm-rag-chatbot/index.html

If you want to run this workload in production, follow instructions here: 
<br/>https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot
