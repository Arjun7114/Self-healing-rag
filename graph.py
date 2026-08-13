# graph.py
# The self-healing loop. Defines the shared STATE and the NODES (steps).
# Each node takes the state, does one job, and returns only what it changed.

from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate

# Reuse what we already built.
from rag import retriever, llm, prompt   # retriever, the LLM, and the answer prompt
from critic import check                  # our grounding critic

MAX_RETRIES = 2  # how many times we'll rewrite + retry before giving up


# 1. THE STATE — the shared object that flows through the whole loop.
#    Every node can read it and update it. This is the loop's memory.
class GraphState(TypedDict):
    question: str            # the CURRENT search query (may get rewritten)
    original_question: str   # the user's original question (never changes)
    documents: List[str]     # the retrieved chunks
    answer: str              # the generated answer
    grounded: bool           # the critic's verdict
    reason: str              # the critic's explanation
    retries: int             # how many times we've retried so far


# 2. THE NODES — one function per step. Each returns a partial state update.

def retrieve(state: GraphState):
    print(f"  [retrieve] searching for: {state['question']}")
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

def generate(state: GraphState):
    print("  [generate] writing an answer...")
    context = "\n\n".join(state["documents"])
    message = prompt.format(context=context, question=state["original_question"])
    answer = llm.invoke(message).content
    return {"answer": answer}

def critique(state: GraphState):
    print("  [critique] fact-checking the answer...")
    context = "\n\n".join(state["documents"])
    verdict = check(context, state["answer"])
    print(f"  [critique] grounded={verdict.grounded} - {verdict.reason}")
    return {"grounded": verdict.grounded, "reason": verdict.reason}

# The rewrite node: when grounded=False, reshape the question into a
# better search query and try again.
rewrite_prompt = ChatPromptTemplate.from_template(
    """The previous search did not find enough information to answer this
question well. Rewrite it as a clearer, more specific search query.
Return ONLY the rewritten query, nothing else.

Original question: {question}"""
)

def rewrite(state: GraphState):
    new_retries = state["retries"] + 1
    print(f"  [rewrite] attempt {new_retries}: improving the query...")
    message = rewrite_prompt.format(question=state["original_question"])
    new_query = llm.invoke(message).content.strip()
    return {"question": new_query, "retries": new_retries}

def give_up(state: GraphState):
    print("  [give_up] no grounded answer found - refusing honestly.")
    return {"answer": "I don't have enough information to answer that."}


# 3. THE GRAPH — wire the nodes together with decision logic.

from langgraph.graph import StateGraph, END

# The decision function: after critique, which way do we go?
def decide(state: GraphState):
    if state["grounded"]:
        return "accept"                       # answer is good -> finish
    if state["retries"] >= MAX_RETRIES:
        return "give_up"                       # out of retries -> refuse
    return "retry"                             # not grounded, retries left -> rewrite

# Build the graph.
workflow = StateGraph(GraphState)

# Register the nodes.
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("critique", critique)
workflow.add_node("rewrite", rewrite)
workflow.add_node("give_up", give_up)

# Set the flow. START -> retrieve -> generate -> critique -> [decision]
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critique")

# The conditional branch out of critique.
workflow.add_conditional_edges(
    "critique",
    decide,
    {
        "accept": END,          # good answer -> stop
        "give_up": "give_up",   # refuse -> then stop
        "retry": "rewrite",     # rewrite the query...
    },
)

# After a rewrite, loop BACK to retrieve -> this is the cycle.
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("give_up", END)

# Compile it into a runnable app.
app = workflow.compile()


# 4. RUN IT.
def run(question: str):
    initial_state = {
        "question": question,
        "original_question": question,
        "retries": 0,
    }
    print(f"\n=== QUESTION: {question} ===")
    result = app.invoke(initial_state)
    print("\n--- FINAL ANSWER ---")
    print(result["answer"])
    return result

if __name__ == "__main__":
    # 1. A question the document CAN answer -> accepts on first try.
    run("How much is the home office stipend?")

    # 2. A question the document CANNOT answer -> should retry, then give up.
    run("What is the CEO's annual salary?")