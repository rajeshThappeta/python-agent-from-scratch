"""A tiny AI agent in plain Python: no framework, a local model through Ollama."""

from ollama import chat

# Sample data, for the demo only.
ORDERS = {"1042": {"status": "Delivered", "date": "2026-11-08"}}


# Step 1: tools are plain functions. The docstring is what the model reads.
def lookup_order(order_id: str) -> str:
    """Return the status and date of one order, given its order id."""
    order = ORDERS.get(order_id)
    if not order:
        return f"No order found with id {order_id}."
    return f"Order {order_id}: {order['status']} on {order['date']}."


def refund_policy() -> str:
    """Return the store's return and refund rules."""
    return "Returns are accepted within 7 days of delivery. Refunds go to the original payment method."


TOOLS = {"lookup_order": lookup_order, "refund_policy": refund_policy}


# Steps 2 to 4: hand the tools to the model, run the loop, add the guards.
def run_agent(question, model="qwen3:14b", max_steps=5):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):  # guard 2: a loop that never ends
        response = chat(model=model, messages=messages, tools=list(TOOLS.values()))
        messages.append(response.message)
        calls = response.message.tool_calls

        if not calls:
            text = response.message.content or ""
            if any(name in text for name in TOOLS):  # guard 1: a call written as text
                messages.append({"role": "user", "content": "Use a real tool call."})
                continue
            return text

        for call in calls:
            name = call.function.name
            try:
                result = TOOLS[name](**call.function.arguments)
            except Exception as error:  # the model reads the error and can recover
                result = f"Tool error: {error}"
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    return "Stopped: too many steps."


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Where is order 1042, and can I return it?"
    print(run_agent(question))
