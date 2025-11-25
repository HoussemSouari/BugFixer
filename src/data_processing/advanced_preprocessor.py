"""
Advanced Java code preprocessor with real variable name preservation.
Includes syntax validation, bug type classification, and code normalization.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import tempfile
import subprocess
import json

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedPair:
    """Preprocessed code pair with metadata"""
    buggy_code: str
    fixed_code: str
    bug_type: str
    buggy_tokens: int
    fixed_tokens: int
    is_valid: bool
    error_message: Optional[str] = None


class JavaSyntaxValidator:
    """Validate Java syntax without running full compilation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.basic_patterns = {
            'unclosed_brace': r'.*\{(?!.*\})',
            'unclosed_bracket': r'.*\[(?!.*\])',
            'unclosed_paren': r'.*\((?!.*\))',
            'mismatched_quotes': r'(?<!\\)(["\'])(?:(?=(\\?))\2.)*?\1',
        }
    
    def is_syntactically_valid(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if Java code is syntactically valid.
        Uses basic heuristics (full compilation not required).
        """
        # Check for basic syntax errors
        if not code.strip():
            return False, "Empty code"
        
        # Count braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            return False, f"Mismatched braces: {open_braces} open, {close_braces} closed"
        
        # Count brackets
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        if open_brackets != close_brackets:
            return False, f"Mismatched brackets: {open_brackets} open, {close_brackets} closed"
        
        # Count parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            return False, f"Mismatched parentheses: {open_parens} open, {close_parens} closed"
        
        # Check for common syntax patterns
        if not self._has_valid_structure(code):
            return False, "Invalid Java structure"
        
        return True, None
    
    def _has_valid_structure(self, code: str) -> bool:
        """Check if code has valid Java structure"""
        # Remove comments
        clean_code = re.sub(r'//.*', '', code)
        clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)
        
        # Should have some content
        if not clean_code.strip():
            return False
        
        # Check for common Java keywords
        java_keywords = ['public', 'private', 'protected', 'class', 'interface', 
                        'void', 'int', 'String', 'boolean', 'for', 'if', 'while']
        has_keyword = any(f'\\b{kw}\\b' in clean_code for kw in java_keywords)
        
        return has_keyword or '{' in clean_code


class AdvancedJavaPreprocessor:
    """
    Advanced preprocessor for Java code that preserves real variable names
    and provides semantic information.
    """
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.validator = JavaSyntaxValidator()
        self.logger = logging.getLogger(__name__)
        
        # Bug-specific prefixes for task-aware training
        self.task_prefixes = {
            'off_by_one': 'Fix off-by-one error:',
            'null_check': 'Add null safety check:',
            'missing_break': 'Add missing break statement:',
            'operator': 'Fix operator error:',
            'logic': 'Fix logic error:',
            'resource_leak': 'Fix resource leak:',
            'type_mismatch': 'Fix type mismatch:',
            'default': 'Fix bug:'
        }
    
    def normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace while preserving string contents"""
        lines = code.split('\n')
        normalized = []
        
        for line in lines:
            # Don't collapse spaces inside strings
            if '"' in line or "'" in line:
                # Just strip leading/trailing
                normalized.append(line.rstrip())
            else:
                # Replace multiple spaces with single space
                line = re.sub(r' +', ' ', line)
                normalized.append(line.rstrip())
        
        # Remove consecutive blank lines
        result = []
        prev_blank = False
        for line in normalized:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            result.append(line)
            prev_blank = is_blank
        
        return '\n'.join(result).strip()
    
    def remove_imports_and_comments(self, code: str, keep_comments: bool = False) -> str:
        """
        Remove imports and optionally comments to focus on method body.
        Preserves class/method signatures.
        """
        lines = code.split('\n')
        result = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip import statements
            if stripped.startswith('import ') or stripped.startswith('package '):
                continue
            
            # Skip comments if not keeping them
            if not keep_comments and (stripped.startswith('//') or stripped.startswith('*')):
                continue
            
            result.append(line)
        
        return '\n'.join(result)
    
    def extract_method_body(self, code: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Extract just the method body for cleaner training data.
        
        Returns:
            Tuple of (body, method_signature, class_signature)
        """
        # Try to find method signature
        method_pattern = r'((?:public|private|protected)?\s*(?:static)?\s*\w+\s+\w+\s*\([^)]*\))'
        method_match = re.search(method_pattern, code)
        method_sig = method_match.group(1) if method_match else None
        
        # Try to find class signature
        class_pattern = r'(class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?)'
        class_match = re.search(class_pattern, code)
        class_sig = class_match.group(1) if class_match else None
        
        # Extract body (everything between first { and last })
        start_idx = code.find('{')
        end_idx = code.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            body = code[start_idx + 1:end_idx].strip()
        else:
            body = code
        
        return body, method_sig, class_sig
    
    def add_task_prefix(self, code: str, bug_type: str = None) -> str:
        """
        Add task-specific prefix based on bug type.
        Helps model understand the task.
        """
        prefix = self.task_prefixes.get(bug_type, self.task_prefixes['default'])
        return f"{prefix} {code}"
    
    def tokenize_quick(self, code: str) -> int:
        """Quick token count approximation for Java"""
        # Split on whitespace and punctuation while respecting strings
        tokens = re.findall(r'\b\w+\b|[{}();,=\[\]]|"[^"]*"|\'[^\']*\'', code)
        return len(tokens)
    
    def is_length_valid(self, code: str) -> bool:
        """Check if code length is within limits"""
        token_count = self.tokenize_quick(code)
        return token_count <= self.max_length
    
    def balance_code_pair(self, buggy: str, fixed: str) -> Tuple[str, str]:
        """
        Balance code pair lengths to be similar.
        Prevents extreme length mismatches.
        """
        buggy_len = self.tokenize_quick(buggy)
        fixed_len = self.tokenize_quick(fixed)
        
        # If one is much longer, truncate it
        if buggy_len > fixed_len * 1.5:
            # Truncate buggy code
            lines = buggy.split('\n')
            while len(lines) > 1 and self.tokenize_quick('\n'.join(lines)) > fixed_len * 1.2:
                lines = lines[:-1]
            buggy = '\n'.join(lines)
        
        return buggy, fixed
    
    def preprocess_pair(
        self,
        buggy_code: str,
        fixed_code: str,
        bug_type: str = None,
        extract_body: bool = True,
        add_prefix: bool = True,
        validate_syntax: bool = True,
    ) -> PreprocessedPair:
        """
        Preprocess a bug-fix pair.
        
        Args:
            buggy_code: Buggy code sample
            fixed_code: Fixed code sample
            bug_type: Type of bug
            extract_body: Whether to extract just method bodies
            add_prefix: Whether to add task prefix
            validate_syntax: Whether to validate syntax
            
        Returns:
            PreprocessedPair object
        """
        try:
            # Step 1: Normalize whitespace
            buggy = self.normalize_whitespace(buggy_code)
            fixed = self.normalize_whitespace(fixed_code)
            
            # Step 2: Remove imports and package statements
            buggy = self.remove_imports_and_comments(buggy)
            fixed = self.remove_imports_and_comments(fixed)
            
            # Step 3: Extract method bodies if requested
            if extract_body:
                buggy_body, _, _ = self.extract_method_body(buggy)
                fixed_body, _, _ = self.extract_method_body(fixed)
                buggy = buggy_body if buggy_body.strip() else buggy
                fixed = fixed_body if fixed_body.strip() else fixed
            
            # Step 4: Check length constraints
            if not self.is_length_valid(buggy) or not self.is_length_valid(fixed):
                return PreprocessedPair(
                    buggy_code=buggy,
                    fixed_code=fixed,
                    bug_type=bug_type or 'unknown',
                    buggy_tokens=self.tokenize_quick(buggy),
                    fixed_tokens=self.tokenize_quick(fixed),
                    is_valid=False,
                    error_message="Code exceeds maximum length"
                )
            
            # Step 5: Balance pair lengths
            buggy, fixed = self.balance_code_pair(buggy, fixed)
            
            # Step 6: Validate syntax
            is_valid = True
            error_msg = None
            
            if validate_syntax:
                buggy_valid, buggy_err = self.validator.is_syntactically_valid(buggy)
                fixed_valid, fixed_err = self.validator.is_syntactically_valid(fixed)
                
                if not buggy_valid or not fixed_valid:
                    is_valid = False
                    error_msg = f"Buggy: {buggy_err}, Fixed: {fixed_err}"
            
            # Step 7: Add task prefix if requested
            if add_prefix and bug_type:
                buggy = self.add_task_prefix(buggy, bug_type)
            
            return PreprocessedPair(
                buggy_code=buggy,
                fixed_code=fixed,
                bug_type=bug_type or 'unknown',
                buggy_tokens=self.tokenize_quick(buggy),
                fixed_tokens=self.tokenize_quick(fixed),
                is_valid=is_valid,
                error_message=error_msg
            )
            
        except Exception as e:
            logger.error(f"Error preprocessing pair: {e}")
            return PreprocessedPair(
                buggy_code=buggy_code,
                fixed_code=fixed_code,
                bug_type=bug_type or 'unknown',
                buggy_tokens=0,
                fixed_tokens=0,
                is_valid=False,
                error_message=str(e)
            )
    
    def preprocess_batch(
        self,
        pairs: List[Tuple[str, str, str]],  # (buggy, fixed, bug_type)
        **kwargs
    ) -> List[PreprocessedPair]:
        """Preprocess a batch of code pairs"""
        results = []
        for buggy, fixed, bug_type in pairs:
            result = self.preprocess_pair(buggy, fixed, bug_type, **kwargs)
            results.append(result)
        return results
    
    def filter_valid_pairs(
        self,
        pairs: List[PreprocessedPair],
        min_buggy_tokens: int = 5,
        min_fixed_tokens: int = 5,
    ) -> List[PreprocessedPair]:
        """Filter to keep only valid, meaningful pairs"""
        valid = []
        for pair in pairs:
            if (pair.is_valid and 
                pair.buggy_tokens >= min_buggy_tokens and 
                pair.fixed_tokens >= min_fixed_tokens and
                pair.buggy_code != pair.fixed_code):
                valid.append(pair)
        
        return valid
    
    def get_statistics(self, pairs: List[PreprocessedPair]) -> Dict[str, Any]:
        """Get statistics about preprocessed pairs"""
        if not pairs:
            return {}
        
        stats = {
            'total_pairs': len(pairs),
            'valid_pairs': sum(1 for p in pairs if p.is_valid),
            'invalid_pairs': sum(1 for p in pairs if not p.is_valid),
            'avg_buggy_tokens': sum(p.buggy_tokens for p in pairs) / len(pairs),
            'avg_fixed_tokens': sum(p.fixed_tokens for p in pairs) / len(pairs),
            'max_buggy_tokens': max(p.buggy_tokens for p in pairs),
            'max_fixed_tokens': max(p.fixed_tokens for p in pairs),
            'min_buggy_tokens': min(p.buggy_tokens for p in pairs),
            'min_fixed_tokens': min(p.fixed_tokens for p in pairs),
        }
        
        # Bug type distribution
        bug_types = {}
        for pair in pairs:
            bug_types[pair.bug_type] = bug_types.get(pair.bug_type, 0) + 1
        stats['bug_type_distribution'] = bug_types
        
        return stats
