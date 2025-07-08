import difflib 


class BugTypeClassifier:
    def __init__(self):
        self.error_types = {
            "syntax_error": "Syntax Error",
            "null_pointer": "Null Pointer Exception",
            "array_index_out_of_bounds": "Array Index Out of Bounds",
            "class_not_found": "Class Not Found",
            "type_mismatch": "Type Mismatch",
            "infinite_loop": "Infinite Loop",
            "other": "Other"
        }

    def classify(self, buggy_code, fixed_code):
        """
        Classify the type of bug based on the difference between buggy and fixed code.
        """
        diff = difflib.ndiff(buggy_code.splitlines(), fixed_code.splitlines())
        changes = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]

        if not changes:
            return self.error_types["other"]

        # Simple heuristic based classification
        if any("null" in change for change in changes):
            return self.error_types["null_pointer"]
        elif any("index" in change for change in changes):
            return self.error_types["array_index_out_of_bounds"]
        elif any("class" in change for change in changes):
            return self.error_types["class_not_found"]
        elif any("type" in change for change in changes):
            return self.error_types["type_mismatch"]
        elif any("loop" in change for change in changes):
            return self.error_types["infinite_loop"]
        
        return self.error_types["syntax_error"]
    

if __name__ == "__main__":
    # Example usage
    classifier = BugTypeClassifier()
    buggy_code = "if (a=b) {return true;}"
    fixed_code = "if (a == b) {return true;}"  # Fixed the assignment operator to equality check
    
    bug_type = classifier.classify(buggy_code, fixed_code)
    print(f"Identified Bug Type: {bug_type}")