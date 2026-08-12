# critic.py
# An independent checker. Given a question, the retrieved context, and an
# answer, it judges whether the answer is actually supported by the context.
# It returns a STRUCTURED verdict (yes/no + reason) our code can act on.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. Define the exact shape we want the verdict to come back in.
#    This is "structured output" — the model must fill these fields,
#    so our code can read .grounded instead of parsing free text.
class Verdict(BaseModel):
    grounded: bool = Field(description="True if the answer is fully supported by the context, else False.")
    reason: str = Field(description="A short explanation of the decision.")

# 2. The critic model. We bind the schema so Ollama returns JSON matching Verdict.
critic_llm = ChatOllama(model="llama3", temperature=0).with_structured_output(Verdict)

# 3. The critic's instructions. Note it judges ONLY grounding, not correctness
#    of style — just: is every claim backed by the context?
critic_prompt = ChatPromptTemplate.from_template(
    """You are a strict fact-checker. Your job is to decide whether the ANSWER
is fully supported by the CONTEXT. 

- If every claim in the answer appears in the context, grounded = true.
- If the answer adds any fact not in the context, grounded = false.
- If the answer says it doesn't know, grounded = true (refusing is safe).

CONTEXT:
{context}

ANSWER:
{answer}

Return your verdict."""
)

def check(context: str, answer: str) -> Verdict:
    message = critic_prompt.format(context=context, answer=answer)
    return critic_llm.invoke(message)

# Quick standalone test.
if __name__ == "__main__":
    context = "Acme provides a one-time home office stipend of 500 dollars."

    good_answer = "The stipend is 500 dollars."
    bad_answer = "The stipend is 1000 dollars and renews yearly."

    print("Testing a GOOD answer:")
    print(check(context, good_answer))

    print("\nTesting a BAD (hallucinated) answer:")
    print(check(context, bad_answer))