"""
Text Summarization Tool
A robust document summarization system using extractive and abstractive methods
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import nltk
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextSummarizer:
    """Main text summarization class supporting multiple algorithms"""
    
    def __init__(self, method='abstractive'):
        """
        Initialize the summarizer
        
        Args:
            method (str): Summarization method - 'abstractive', 'lsa', 'luhn', or 'textrank'
        """
        self.method = method.lower()
        self.summarizer = None
        
        if self.method == 'abstractive':
            print("Loading abstractive model (this may take a moment)...")
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        elif self.method in ['lsa', 'luhn', 'textrank']:
            print(f"Using {self.method.upper()} extractive summarization")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def summarize(self, text: str, sentences_count: int = 3, 
                  max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize the input text
        
        Args:
            text (str): Input text to summarize
            sentences_count (int): Number of sentences for extractive methods
            max_length (int): Maximum length for abstractive summary
            min_length (int): Minimum length for abstractive summary
            
        Returns:
            str: Summarized text
        """
        if not text or len(text.strip()) == 0:
            return "Error: Empty text provided"
        
        if self.method == 'abstractive':
            return self._abstractive_summarize(text, max_length, min_length)
        else:
            return self._extractive_summarize(text, sentences_count)
    
    def _abstractive_summarize(self, text: str, max_length: int, min_length: int) -> str:
        """Generate abstractive summary using transformer model"""
        try:
            # Split long texts into chunks if needed
            max_input = 1024
            if len(text.split()) > max_input:
                chunks = self._split_text(text, max_input)
                summaries = []
                for chunk in chunks:
                    result = self.summarizer(chunk, max_length=max_length, 
                                            min_length=min_length, do_sample=False)
                    summaries.append(result[0]['summary_text'])
                return ' '.join(summaries)
            else:
                result = self.summarizer(text, max_length=max_length, 
                                        min_length=min_length, do_sample=False)
                return result[0]['summary_text']
        except Exception as e:
            return f"Error during summarization: {str(e)}"
    
    def _extractive_summarize(self, text: str, sentences_count: int) -> str:
        """Generate extractive summary using statistical methods"""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            
            if self.method == 'lsa':
                summarizer = LsaSummarizer()
            elif self.method == 'luhn':
                summarizer = LuhnSummarizer()
            else:  # textrank
                summarizer = TextRankSummarizer()
            
            summary = summarizer(parser.document, sentences_count)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            return f"Error during summarization: {str(e)}"
    
    def _split_text(self, text: str, max_words: int) -> list:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_words:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def read_file(filepath: str) -> Tuple[bool, str]:
    """
    Read text from file
    
    Args:
        filepath (str): Path to input file
        
    Returns:
        Tuple[bool, str]: Success status and text content or error message
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return False, f"File not found: {filepath}"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return True, content
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def save_summary(summary: str, output_path: Optional[str] = None) -> bool:
    """
    Save summary to file
    
    Args:
        summary (str): Summary text to save
        output_path (str, optional): Output file path
        
    Returns:
        bool: Success status
    """
    try:
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\nSummary saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False


def main():
    """Main function to run the summarization tool"""
    parser = argparse.ArgumentParser(
        description='Text Summarization Tool - Automatic document summarization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i document.txt
  %(prog)s -i document.txt -o summary.txt -m lsa -s 5
  %(prog)s -t "Your text here" -m abstractive
        """
    )
    
    parser.add_argument('-i', '--input', type=str, help='Input file path')
    parser.add_argument('-o', '--output', type=str, help='Output file path (optional)')
    parser.add_argument('-t', '--text', type=str, help='Direct text input (alternative to -i)')
    parser.add_argument('-m', '--method', type=str, default='abstractive',
                       choices=['abstractive', 'lsa', 'luhn', 'textrank'],
                       help='Summarization method (default: abstractive)')
    parser.add_argument('-s', '--sentences', type=int, default=3,
                       help='Number of sentences for extractive methods (default: 3)')
    parser.add_argument('--max-length', type=int, default=150,
                       help='Max length for abstractive summary (default: 150)')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Min length for abstractive summary (default: 50)')
    
    args = parser.parse_args()
    
    # Get input text
    text = None
    if args.input:
        success, result = read_file(args.input)
        if not success:
            print(f"Error: {result}")
            sys.exit(1)
        text = result
    elif args.text:
        text = args.text
    else:
        print("Error: Please provide input using -i (file) or -t (text)")
        parser.print_help()
        sys.exit(1)
    
    # Initialize summarizer
    try:
        summarizer = TextSummarizer(method=args.method)
    except Exception as e:
        print(f"Error initializing summarizer: {str(e)}")
        sys.exit(1)
    
    # Generate summary
    print("\nGenerating summary...")
    summary = summarizer.summarize(
        text,
        sentences_count=args.sentences,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Display results
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary)
    print("="*70)
    
    # Save if output path provided
    if args.output:
        save_summary(summary, args.output)


if __name__ == "__main__":
    main()