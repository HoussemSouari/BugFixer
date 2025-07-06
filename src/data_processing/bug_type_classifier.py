class BugTypeClassifier:
    ERROR_TYPES = ["SYNTAX", "LOGICAL", "RUNTIME", "OTHER"]
    
    def classify(self, buggy_code: str, fixed_code: str) -> str:
        """
        Classify bug type based on code differences
        - Syntax: Punctuation changes
        - Logical: Operator/conditional changes
        - Runtime: Exception handling added
        - Other: Complex refactoring
        """
        # Check for added exception handling (Runtime)
        if self._has_runtime_fix(buggy_code, fixed_code):
            return "RUNTIME"
        
        # Check for operator changes (Logical)
        if self._has_logical_error(buggy_code, fixed_code):
            return "LOGICAL"
        
        # Check for punctuation changes (Syntax)
        if self._has_syntax_error(buggy_code, fixed_code):
            return "SYNTAX"
        
        return "OTHER"
    
    def _has_runtime_fix(self, buggy: str, fixed: str) -> bool:
        """Check if exception handling was added"""
        runtime_keywords = ['try', 'catch', 'finally', 'throws', 'throw']
        return any(
            kw in fixed and kw not in buggy 
            for kw in runtime_keywords
        )
    
    def _has_logical_error(self, buggy: str, fixed: str) -> bool:
        """Check for logical operator changes"""
        operators = ['+', '-', '*', '/', '%', '&&', '||', '==', '!=', '<', '>', '<=', '>=']
        logical_keywords = ['if', 'else', 'for', 'while', 'switch', 'case']
        
        # Check for operator changes
        for op in operators:
            if (op in fixed) != (op in buggy):
                return True
        
        # Check for conditional changes
        for kw in logical_keywords:
            if (kw in fixed) != (kw in buggy):
                return True
                
        return False
    
    def _has_syntax_error(self, buggy: str, fixed: str) -> bool:
        """Check for punctuation mismatch"""
        punctuation = [';', '{', '}', '(', ')', '[', ']']
        buggy_count = sum(buggy.count(p) for p in punctuation)
        fixed_count = sum(fixed.count(p) for p in punctuation)
        return buggy_count != fixed_count