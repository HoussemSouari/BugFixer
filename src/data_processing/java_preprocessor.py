import re 
import ast 
from tree_sitter_languages import Language
from tree_sitter import Parser


class JavaPreprocessor:
    """
    Java code preprocessor for cleaning and normalizing code
    """

    def __init__(self):
        """
        Initialize the Java preprocessor
        """
        self.parser = self._initialize_parser()

    def _initialize_parser(self):
        """Build and configure tree-sitter parser"""
        try:
            JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
            parser = Parser()
            parser.language = JAVA_LANGUAGE
            return parser
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Java parser: {str(e)}")

    def clean_code(self, code: str) -> str:
        """
        Clean Java code by removing comments, extra whitespace, and normalizing syntax

        Args:
            code: Raw Java code string

        Returns:
            Cleaned Java code string
        """
        # Remove comments
        code = re.sub(r'//.*|/\*.*?\*/', '', code, flags=re.DOTALL)

        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code).strip()

        # Parse and reformat using tree-sitter
        tree = self.parser.parse(bytes(code, "utf8"))
        return self._walk_tree(tree.root_node)

    def _walk_tree(self, node):
        """Recursively walk the AST to extract cleaned code"""
        if node.is_named:
            return node.text.decode('utf-8')
        return ''