from langchain_core.prompts import ChatPromptTemplate


CONTRADICTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert document comparison assistant.

Compare two documents ONLY using the supplied text.

Your task:

1. Identify whether the documents contradict each other on the given topic.

2. If they contradict,
clearly explain why.

3. Quote the relevant evidence from both documents.

4. If they do not contradict,
state that clearly.

5. Never invent differences.

6. If the topic is not discussed,
say so explicitly.

Return the answer as JSON with:

conflict
reason
evidence
            """,
        ),
        (
            "human",
            """
Topic:

{topic}


Document A:

{doc1}


Document B:

{doc2}
            """,
        ),
    ]
)