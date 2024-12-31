from src.db import vector_db, load_to_db
from src.models import hf_embeddings, code_llama
from src.chain import response_chain






if __name__ == '__main__':
    repo_path = "/Users/dengkai/workspace/machine-learning/llm-project/knowledge-management/repochat/cloned_repo/knowledge-management"
    code = load_to_db(repo_path)
    embedding = hf_embeddings()
    vector_db = vector_db(embedding, code)
    llm = code_llama()
    conversational_retrieval_chain = response_chain(vector_db, llm)

    prompt = "how to use response chain?"
    result = conversational_retrieval_chain(prompt)
    print(result)
