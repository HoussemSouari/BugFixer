from tree_sitter import Language, Parser
from .java_preprocessor import clean_java_code
import os
import re

class JavaTokenizer:
    """
    Java code tokenizer using tree-sitter for syntax-aware tokenization
    
    Features:
    - Preserves code structure
    - Handles Java-specific syntax
    - Recursively parses AST
    - Optionally uses preprocessing
    """
    
    def __init__(self, use_preprocessing=True):
        """
        Initialize Java tokenizer
        
        Args:
            use_preprocessing: Apply code cleaning before tokenization
        """
        self.use_preprocessing = use_preprocessing
        self.parser = self._initialize_parser()
        
    def _initialize_parser(self):
        """Build and configure tree-sitter parser"""
        try:
            # Load language library
            JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
            parser = Parser()
            parser.set_language(JAVA_LANGUAGE)
            return parser
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Java parser: {str(e)}")
    
    def tokenize(self, code: str) -> list:
        """
        Tokenize Java code while preserving language structure
        
        Args:
            code: Java code string
            
        Returns:
            List of tokens
        """
        # Preprocess if enabled
        if self.use_preprocessing:
            code = clean_java_code(code)
        
        # Parse code
        tree = self.parser.parse(bytes(code, "utf8"))
        
        # Traverse AST to extract tokens
        return self._walk_tree(tree.root_node)
    
    def _walk_tree(self, node):
        """
        Recursively traverse AST to extract tokens
        
        Args:
            node: Current AST node
            
        Returns:
            List of tokens from this node and its children
        """
        tokens = []
        
        # Add current node if it's a leaf (has no children)
        if len(node.children) == 0 and node.text:
            token = node.text.decode('utf8')
            
            # Split compound tokens (e.g., "int x=5;" -> ["int", "x", "=", "5", ";"])
            if re.match(r'^\w+[=;,:(){}\[\]]', token):
                tokens.extend(self._split_compound_token(token))
            else:
                tokens.append(token)
        
        # Process children recursively
        for child in node.children:
            tokens.extend(self._walk_tree(child))
            
        return tokens
    
    def _split_compound_token(self, token: str) -> list:
        """
        Split compound tokens that contain multiple logical tokens
        
        Args:
            token: String token that may need splitting
            
        Returns:
            List of split tokens
        """
        # Define Java token boundaries
        patterns = [
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # Identifiers
            r'0[xX][0-9a-fA-F]+',        # Hex literals
            r'\d+\.\d+',                  # Floating point
            r'\d+',                       # Integers
            r'!=|==|<=|>=|&&|\|\|',      # Operators
            r'[+\-*/%&|^<>=!]',          # Single char operators
            r'[();,:{}\[\]\.]',           # Punctuation
        ]
        
        token_re = re.compile(r'(' + '|'.join(patterns) + r')')
        return [t for t in token_re.split(token) if t and not t.isspace()]


def build_language_library():
    """Build tree-sitter language library (run once)"""
    # Create build directory if not exists
    os.makedirs('build', exist_ok=True)
    
    # Clone tree-sitter-java if not available
    if not os.path.exists('vendor/tree-sitter-java'):
        os.makedirs('vendor', exist_ok=True)
        print("Cloning tree-sitter-java repository...")
        os.system('git clone https://github.com/tree-sitter/tree-sitter-java vendor/tree-sitter-java')
    
    # Build language library
    Language.build_library(
        'build/my-languages.so',
        ['vendor/tree-sitter-java']
    )
    print("Successfully built tree-sitter languages")


class RegexTokenizer:
    """Alternative regex-based tokenizer for simplicity"""
    
    @staticmethod
    def tokenize(code: str) -> list:
        """
        Tokenize Java code using regex patterns
        
        Args:
            code: Java code string
            
        Returns:
            List of tokens
        """
        # Define Java token patterns
        patterns = [
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # Identifiers
            r'0[xX][0-9a-fA-F]+',        # Hex literals
            r'\d+\.\d+',                  # Floating point
            r'\d+',                       # Integers
            r'!=|==|<=|>=|&&|\|\|',      # Operators
            r'[+\-*/%&|^<>=!]',          # Single char operators
            r'[();,:{}\[\]\.]',           # Punctuation
            r'"(?:\\.|[^\\"])*"',         # Strings
            r"'(?:\\.|[^\\'])*'"          # Chars
        ]
        
        token_re = re.compile(r'(' + '|'.join(patterns) + r')|\s+')
        tokens = [t for t in token_re.split(code) if t and not t.isspace()]
        return tokens


# Run this once to build the language library
# if __name__ == "__main__":
#     build_language_library()