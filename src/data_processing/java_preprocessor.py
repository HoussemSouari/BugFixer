import re 
import ast 
from tree_sitter import Language, Parser


class JavaPreprocessor:
    def __init__(self):
        self.parser = self._initialize_parser()

    def _initialize_parser(self):

        try:
            Language.build_library(
                'build/my-languages.so',
                ['vendor/tree-sitter-java']
            )

            JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
            parser = Parser()
            parser.set_language(JAVA_LANGUAGE)
            return parser
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Java parser: {str(e)}")
        
    def clean_code(self, code:str) -> str :

        try : 
            code, str_replacements = self._preserve_string_literals(code)

            code = self._remove_comments(code)

            code = self._normalize_generics_annotations(code)

            code = self._standardize_formatting(code)

            code = self._restore_string_literals(code, str_replacements)

            if not self._is_valid_java(code): 
                return ValueError ("Invalid Java code after preprocessing")
            
            return code
        except Exception as e :
            raise RuntimeError(f"Preprocessing failed: {str(e)}")


    def _preserve_string_literals(self, code: str) -> tuple[str, dict]:
        """Replace string literals with placeholders"""
        replacements = {}
        str_count = 0
        
        def replace_str(match):
            nonlocal str_count
            content = match.group(0)
            placeholder = f"__STR_{str_count}__"
            replacements[placeholder] = content
            str_count += 1
            return placeholder
        
        # Replace string literals
        code = re.sub(r'"(?:\\.|[^\\"])*"', replace_str, code)
        # Replace character literals
        code = re.sub(r"'(?:\\.|[^\\'])*'", replace_str, code)
        
        return code, replacements
    
    def _remove_comments(self, code: str) -> str:
        """Remove all comments from code"""
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code
    
    def _normalize_generics_annotations(self, code: str) -> str:
        """Normalize Java-specific constructs"""
        # Handle generics
        code = re.sub(r'<', ' < ', code)
        code = re.sub(r'>', ' > ', code)
        # Handle annotations
        code = re.sub(r'@(\w+)', r'@ \1', code)
        return code
    
    def _standardize_formatting(self, code: str) -> str:
        """Standardize code formatting"""
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        # Format control structures
        keywords = ['if', 'for', 'while', 'try', 'catch', 'switch']
        for kw in keywords:
            code = re.sub(rf'\b{kw}\s*\(', f'{kw} (', code)
        return code.strip()
    
    def _restore_string_literals(self, code: str, replacements: dict) -> str:
        """Restore original string literals"""
        for placeholder, original in replacements.items():
            code = code.replace(placeholder, original)
        return code
    
    def _is_valid_java(self, code: str) -> bool:
        """Validate Java code structure using tree-sitter"""
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return not tree.root_node.has_error
        except:
            return False