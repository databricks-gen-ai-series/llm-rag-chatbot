# Databricks PDF-based Retrieval Augmented Generation (RAG) Proof of Concept (PoC) Repo

Author: Han Zhang, Ajinkya Gutti, Z Sun, Tian Tan, Solutions Architects at Databricks. 

Contact: h.zhang@databricks.com

**_Updated in August 2024_**

DISCLAIMER: PLEASE REVIEW LICENSING INFORMATION HERE BEFORE PROCEEDING. 

MIT License
Copyright (c) 2024 HZ-SA-DSAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Intro
Welcome to the Databricks PDF-based Retrieval Augmented Generation (RAG) Proof of Concept (PoC) repository. This repository is designed to guide Databricks users through the creation of a fast and efficient RAG PoC based on PDFs that you already have, utilizing the necessary Databricks infrastructure components. Inspired by the solution accelerator found at [Databricks Lakehouse AI: Deploy Your LLM Chatbot](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot?itm_data=demo_center), this repository brings modifications for enhanced usability, flexibility, and simplicity.

## Overview
This repository enables users to learn how to create and deploy a real-time Q&A chatbot using Databricks' RAG and serverless capabilities. By leveraging the Meta Llama 3.1 70B Instruct Foundation Model, users can expect smart, contextually relevant responses. This repo is highly flexible; the underlying embedding and chat model can be easily swapped out for other models to meet specific needs.

RAG is a sophisticated technique that enriches the Large Language Model (LLM) prompt with additional, domain-specific context, enabling the model to deliver more accurate answers. This method yields impressive results using public models, eliminating the need for deploying and fine-tuning proprietary LLMs.

## Features

1. Flexibility in Parameters: All important parameters can now be adjusted at the beginning of each notebook, offering significant flexiblity and customization

2. Focused Content: The repo is divided into Step 1 and Step 2 notebooks, which should be run in sequence. There are no other steps required, perfectly suitable for a workshop under 2 hours. 

3. Custom Testing: Users are encouraged to upload their own PDFs to the UC volume to test the solution's effectiveness.

4. Latest Technologies from Databricks: This repo leverages Databricks Mosaic AI Agent Framework and Agent Evaluation to build Production-quality Agentic and Retrieval Augmented Generation Apps. 

## Learning Objectives

By utilizing this repository, you will gain practical experience in:

1. Preparing Clean Documents: Learn how to prepare and clean documents to build an internal knowledge base, specializing your chatbot for your specific domain.

2. Leveraging Databricks Vector Search: Utilize our Foundation Model endpoint to create and store document embeddings, enhancing the chatbot's ability to retrieve relevant information.

3. Searching Similar Documents: Employ Databricks Vector Search to find similar documents within your knowledge database, ensuring your chatbot can pull from a rich source of information.

4. Deploying Real-Time Models: Discover how to deploy a real-time model using RAG, providing augmented context in the prompt for more accurate responses.

5. Utilizing Meta Llama 3.1 Model: Take advantage of the fully managed Databricks Foundation Model endpoint to ensure your chatbot benefits from the latest advancements in AI.

## Getting Started
To begin, ensure you follow the notebooks in the prescribed order:

Step 1 Notebook: Set up your environment and prepare your documents for ingestion, chunking, and embedding generation.

Step 2 Notebook: Build conversation chains with langchain, vector search, and execute queries. Conduct Evaluation experiment using LLM as a judge with Mosaic AI Agent Evaluation.

## Summary
This repository offers a comprehensive guide to deploying a real-time Q&A chatbot using Databricks' cutting-edge RAG and serverless technologies. By following the steps outlined, users will not only enhance their understanding of these technologies but also create a specialized chatbot capable of delivering smart, contextually relevant responses.