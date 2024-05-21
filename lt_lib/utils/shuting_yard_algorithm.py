# Author: Loïc Thiriet


# Function to determine the priority of an operator
def priority(operator):
    if operator == "+" or operator == "-":
        return 1
    elif operator == "*" or operator == "/":
        return 2
    return 0


# Function to apply an operator to the values in the queue
def apply_operator(operator_stack, values_queue):
    operator = operator_stack.pop()
    b = values_queue.pop()
    a = values_queue.pop()

    if operator == "+":
        values_queue.append(a + b)
    elif operator == "-":
        values_queue.append(a - b)
    elif operator == "*":
        values_queue.append(a * b)
    elif operator == "/":
        values_queue.append(a / b)

    return operator_stack, values_queue


# Custom implementation of the Shunting Yard algorithm to evaluate mathematical expressions
def shunting_yard_custom(expression: str, mapping_metrics_to_values: dict[str, float | int] = {}):
    # Remove spaces from the expression
    expression = expression.replace(" ", "")
    # Initialize stacks for values and operators
    values_queue = []
    operator_stack = []

    i = 0
    # Iterate through each character in the expression
    while i < len(expression):
        token = expression[i]

        # If the token is a digit, extract the entire number
        if token.isdigit():
            j = i + 1
            while j < len(expression) and (expression[j].isdigit() or expression[j] == "."):
                j += 1
            values_queue.append(float(expression[i:j]))
            i = j
        # If the token is "[" followed by metric characters, extract the metric value from the mapping
        elif token == "[":
            j = i + 1
            while j < len(expression) and (expression[j].isalnum() or expression[j] in "._-"):
                j += 1
            # Add the mapped metric value to the values queue
            values_queue.append(float(mapping_metrics_to_values[expression[i + 1 : j]]))
            # Increment i to skip the "]" character
            i = j + 1
        # If the token is an operator, handle the shunting yard algorithm
        elif token in "+-*/":
            while operator_stack and priority(operator_stack[-1]) >= priority(token):
                operator_stack, values_queue = apply_operator(operator_stack, values_queue)
            operator_stack.append(token)
            i += 1
        # If the token is "(", push it onto the operator stack
        elif token == "(":
            operator_stack.append(token)
            i += 1
        # If the token is ")", handle operators until an opening parenthesis is encountered
        elif token == ")":
            while operator_stack[-1] != "(":
                operator_stack, values_queue = apply_operator(operator_stack, values_queue)
            # Pop the opening parenthesis from the operator stack
            operator_stack.pop()
            i += 1
        # If the token is none of the above, move to the next character
        else:
            i += 1

    # Handle any remaining operators in the stacks
    while operator_stack:
        operator_stack, values_queue = apply_operator(operator_stack, values_queue)

    # The final result is the only value left in the values queue
    return values_queue[0]


if __name__ == "__main__":
    # Example usage:
    mapping = {"a.b1": 2, "b.c": 5}
    expression = "(([b.c] + 4) - 1) / [a.b1]"
    result = shunting_yard_custom(expression, mapping)
    print(f"Result: {result}")
