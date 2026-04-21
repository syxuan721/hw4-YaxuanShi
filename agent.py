"""Math agent that solves questions using tools in a ReAct loop."""

import json
import time

from dotenv import load_dotenv
from pydantic_ai import Agent
from calculator import calculate

load_dotenv()

MODEL = "google-gla:gemini-2.5-flash"

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step. "
        "Use the calculator tool for arithmetic. "
        "Use the product_lookup tool when a question mentions products from the catalog. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return calculate(expression)


@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name."""
    with open("products.json", "r", encoding="utf-8") as f:
        products = json.load(f)

    if product_name in products:
        return str(products[product_name])

    available_products = ", ".join(products.keys())
    return f"Product not found. Available products: {available_products}"


def load_questions(path: str = "math_questions.md") -> list[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def run_question(question_number: int, question: str):
    print(f"## Question {question_number}")
    print(f"> {question}\n")

    last_error = None
    for attempt in range(3):
        try:
            result = agent.run_sync(question)
            break
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("Retrying in 5 seconds...\n")
                time.sleep(5)
    else:
        raise last_error

    print("### Trace")
    for message in result.all_messages():
        for part in message.parts:
            kind = part.part_kind
            if kind in ("user-prompt", "system-prompt"):
                continue
            elif kind == "text":
                print(f"- **Reason:** {part.content}")
            elif kind == "tool-call":
                print(f"- **Act:** `{part.tool_name}({part.args})`")
            elif kind == "tool-return":
                print(f"- **Result:** `{part.content}`")

    print(f"\n**Answer:** {result.output}\n")
    print("---\n")


def main():
    questions = load_questions()

    #selecte
    selected = [1, 5]

    for q_num in selected:
        run_question(q_num, questions[q_num - 1])


if __name__ == "__main__":
    main()
