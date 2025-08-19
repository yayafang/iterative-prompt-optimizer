import subprocess
import tempfile
import traceback
import re
import logging
from typing import List, NamedTuple
import sys
import os


class ExecuteResult(NamedTuple):
    is_passing: bool
    passed_tests: List[str]
    failed_tests: List[str]


class Executor:
    def __init__(self, model, temperature: float, seed: int):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        logging.debug(
            f"Executor initialized with temperature: {temperature}, seed: {seed}"
        )

    def _extract_actual_output(self, stdout: str, stderr: str) -> str:
        logging.info(
            f"Attempting to extract output from:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
        match = re.search(r"RESULT:\s*(.+)$", stdout, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        logging.warning("No 'RESULT:' found in output")
        if stderr.strip():
            return f"Error: {stderr.strip()}"
        return stdout.strip() if stdout.strip() else "No output"

    def _preprocess_tests(
        self, tests: List[str], func: str
    ) -> tuple[List[tuple[str, str, str]], str]:
        """
        Returns: (list of (prepared_call, expected_output, original_test), aligned_function)
        """
        func = re.sub(r"```python\s*", "", func)
        func = re.sub(r"```\s*$", "", func)
        func = func.strip("`").strip()

        # Extract function name from the first test as reference
        pattern = re.compile(r"assert\s+(\w+)\(")
        reference_func_name = None

        # Get reference function name from first valid test
        for test in tests:
            if match := pattern.search(test):
                reference_func_name = match.group(1)
                break

        if not reference_func_name:
            raise ValueError("Could not find function name in test cases")

        # Extract current function name from provided function code
        func_def_pattern = re.compile(r"def\s+(\w+)\s*\(")
        current_func_match = func_def_pattern.search(func)
        if not current_func_match:
            raise ValueError("Could not find function definition in provided code")

        current_func_name = current_func_match.group(1)

        # Replace function name in code if different
        if current_func_name != reference_func_name:
            func = func.replace(
                f"def {current_func_name}", f"def {reference_func_name}", 1
            )

        # Process and validate each test
        processed_tests = []
        for test in tests:
            test_type = self._classify_test_type(test)
            prepared_call, expected_output, original_test = (
                self._validate_and_prepare_test(test, test_type)
            )
            processed_tests.append((prepared_call, expected_output, original_test))

        return processed_tests, func

    def _classify_test_type(self, test: str) -> str:
        """Classify test based on input and output types"""
        # Extract input and output parts
        match = re.match(r"assert\s+\w+\((.*?)\)\s*==\s*(.+)$", test.strip())
        if not match:
            raise ValueError(f"Invalid test format: {test}")

        inputs, output = match.groups()

        # Classify input type
        input_type = self._get_argument_type(inputs)
        # Classify output type
        output_type = self._get_value_type(output)

        return f"{input_type}_to_{output_type}"

    def _get_argument_type(self, arg_str: str) -> str:
        """Determine type of argument(s)"""
        # Remove all whitespace for consistent parsing
        arg_str = arg_str.strip()

        if arg_str.startswith("["):  # List
            if "'" in arg_str or '"' in arg_str:  # List of strings
                return "list_string"
            return "list_numeric"
        elif arg_str.startswith("("):  # Tuple
            return "tuple"
        elif arg_str.startswith('"') or arg_str.startswith("'"):  # String
            return "string"
        elif "," in arg_str:  # Multiple arguments
            return "multiple"
        else:  # Single numeric
            return "numeric"

    def _get_value_type(self, val_str: str) -> str:
        """Determine type of expected output"""
        val_str = val_str.strip()

        if val_str in ("True", "False"):
            return "boolean"
        elif val_str.startswith("["):
            if "'" in val_str or '"' in val_str:
                return "list_string"
            return "list_numeric"
        elif val_str.startswith("("):
            return "tuple"
        elif val_str.startswith('"') or val_str.startswith("'"):
            return "string"
        else:
            return "numeric"

    def _validate_and_prepare_test(
        self, test: str, test_type: str
    ) -> tuple[str, str, str]:
        """
        Validates and prepares test for execution.
        Returns: (prepared_func_call, expected_output, original_test)
        """

        # Extract function name and args string
        func_call_match = re.match(r"assert\s+(\w+)\((.*?)\)\s*==", test)
        expected_match = re.search(r"==\s*(.+)$", test)

        if not func_call_match or not expected_match:
            raise ValueError(f"Invalid test format: {test}")

        func_name = func_call_match.group(1)
        args_str = func_call_match.group(2)
        expected_output = expected_match.group(1).strip()

        # Process arguments based on type
        if test_type.startswith("tuple"):
            processed_args = self._split_tuple_args(args_str)
            # Create valid Python code for function call
            prepared_call = f"{func_name}({', '.join(processed_args)})"

        elif test_type.startswith("list_string"):
            processed_args = self._split_nested_list_args(args_str)
            prepared_call = f"{func_name}({', '.join(processed_args)})"

        elif "string" in test_type:
            processed_args = self._split_string_args(args_str)
            prepared_call = f"{func_name}({', '.join(processed_args)})"

        else:
            # Default handling for simple types
            prepared_call = f"{func_name}({args_str})"

        return prepared_call, expected_output, test

    def _split_tuple_args(self, args_str: str) -> list:
        """
        Handle tuple arguments with proper nested structure preservation
        Args:
            args_str: String containing tuple arguments
        Returns:
            List containing properly formatted tuple arguments
        """
        # Remove any outer whitespace
        args_str = args_str.strip()

        if not args_str:
            return []

        # If it's a single tuple argument, return as is
        if (
            args_str.startswith("(")
            and args_str.endswith(")")
            and args_str.count("(") == args_str.count(")")
        ):
            return [args_str]

        # For multiple tuple arguments
        paren_level = 0
        current_arg = []
        args = []
        quote_char = None  # Track if we're inside a string

        for i, char in enumerate(args_str):
            # Handle string quotes
            if char in "\"'":
                if quote_char is None:  # Start of string
                    quote_char = char
                elif char == quote_char:  # End of string
                    quote_char = None

            # Skip processing special chars if we're inside a string
            if quote_char is not None:
                current_arg.append(char)
                continue

            if char == "(":
                paren_level += 1
                current_arg.append(char)
            elif char == ")":
                paren_level -= 1
                current_arg.append(char)

                # If we've closed a complete tuple and hit a comma or end
                if paren_level == 0 and (
                    i + 1 >= len(args_str) or args_str[i + 1] == ","
                ):
                    args.append("".join(current_arg).strip())
                    current_arg = []
            elif char == "," and paren_level == 0:
                # Skip the comma between tuples
                continue
            else:
                current_arg.append(char)

        # Add any remaining argument
        if current_arg:
            args.append("".join(current_arg).strip())

        return args

    def _split_nested_list_args(self, args_str: str) -> list:
        """Handle nested list arguments"""
        # Example: [['green', 'orange'], ['black', 'white']]
        bracket_level = 0
        current_arg = []
        args = []

        for char in args_str:
            if char == "[":
                bracket_level += 1
                current_arg.append(char)
            elif char == "]":
                bracket_level -= 1
                current_arg.append(char)
                if bracket_level == 0:
                    args.append("".join(current_arg).strip())
                    current_arg = []
            elif char == "," and bracket_level == 0:
                continue
            else:
                current_arg.append(char)

        return args

    def _split_string_args(self, args_str: str) -> list:
        """
        Handle string arguments with mixed quote styles
        Input: '"Python", "PHP", "Java"'
        Output: ['"Python"', '"PHP"', '"Java"']
        """
        # Example: '"Python", "PHP", "Java"' or 'hello', 'world'
        args = []
        quote_chars = {'"', "'"}
        current_arg = []
        in_quotes = False
        current_quote = None

        for char in args_str:
            if char in quote_chars and not in_quotes:
                # Start of a quoted string
                in_quotes = True
                current_quote = char
                current_arg.append(char)
            elif char == current_quote and in_quotes:
                # End of a quoted string
                in_quotes = False
                current_arg.append(char)
                if not any(c.isspace() or c == "," for c in "".join(current_arg)):
                    args.append("".join(current_arg))
                    current_arg = []
            elif char == "," and not in_quotes:
                # End of an argument
                if current_arg:
                    args.append("".join(current_arg).strip())
                    current_arg = []
            else:
                current_arg.append(char)

        # Add the last argument if there is one
        if current_arg:
            args.append("".join(current_arg).strip())

        return [arg.strip() for arg in args if arg.strip()]

    def execute(self, func: str, tests: List[str], timeout: int = 80) -> ExecuteResult:

        # Preprocess tests and align function name
        try:
            processed_tests, aligned_func = self._preprocess_tests(tests, func)
        except ValueError as e:
            return ExecuteResult(False, [], [f"Preprocessing error: {str(e)}"])

        imports = "from typing import *"
        all_passed_tests = []
        all_failed_tests = []
        is_passing = True

        for i, (prepared_call, expected_output, original_test) in enumerate(
            processed_tests, 1
        ):
            temp_file_name = None
            try:
                test_script = f"""{imports}
{aligned_func}

try:
    result = {prepared_call}
    print(f"RESULT: {{result}}")
    assert result == {expected_output}
except Exception as e:
    print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{str(e)}}")
"""

                # Create temporary file for test execution
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as temp_file:
                    temp_file_name = temp_file.name
                    temp_file.write(test_script)

                # Run the test
                result = subprocess.run(
                    ["python", temp_file_name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if result.returncode == 0:
                    all_passed_tests.append(prepared_call)
                else:
                    # 1. Execution failure
                    is_passing = False
                    actual_output = self._extract_actual_output(
                        result.stdout, result.stderr
                    )
                    error_output = result.stderr.strip()
                    error_lines = error_output.split("\n")

                    # Extract error details from stderr
                    if len(error_lines) > 2:
                        error_type = error_lines[-2]  # e.g., "TypeError", "ValueError"
                        error_message = error_lines[-1]  # Actual error message
                    else:
                        error_type = "Unknown Error"
                        error_message = error_output

                    failure_info = f"""
Function call: {prepared_call}
Expected output: {expected_output}
Actual output: {actual_output}
Error type: {error_type}
Error message: {error_message}
"""
                    all_failed_tests.append(failure_info.strip())

            # 2. Timeout failures
            except subprocess.TimeoutExpired:
                is_passing = False
                failure_info = f"""
Function call: {prepared_call}
Error: TIMEOUT - Execution took longer than {timeout} seconds
"""
                all_failed_tests.append(failure_info.strip())
                logging.info(f"Test {i}/{len(tests)} timed out")

            # 3. Other exceptions
            except Exception as e:
                is_passing = False
                error_type = type(e).__name__
                error_message = str(e)
                tb = traceback.extract_tb(sys.exc_info()[2])
                if tb:
                    file, line, func, text = tb[-1]
                    error_location = f"in function '{func}', line {line}"
                else:
                    error_location = "unknown location"

                failure_info = f"""
Function call: {prepared_call}
Error type: {error_type}
Error message: {error_message}
Error location: {error_location}
"""
                all_failed_tests.append(failure_info.strip())
                logging.info(
                    f"Error in test {i}/{len(tests)}: {error_type} - {error_message} at {error_location}"
                )

            finally:
                if temp_file_name and os.path.exists(temp_file_name):
                    os.unlink(temp_file_name)  # Delete temp file

        return ExecuteResult(is_passing, all_passed_tests, all_failed_tests)
