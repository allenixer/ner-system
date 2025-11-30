# Text Summarization Tool

A robust and versatile document summarization system that supports both extractive and abstractive summarization methods. Perfect for quickly digesting long documents, articles, reports, and more.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Multiple Summarization Methods**
  - **Abstractive**: Uses state-of-the-art transformer models (BART) to generate human-like summaries
  - **LSA**: Latent Semantic Analysis for extractive summarization
  - **Luhn**: Frequency-based extractive summarization
  - **TextRank**: Graph-based extractive summarization

- **Flexible Input Options**
  - Read from text files
  - Direct text input via command line
  - Batch processing support

- **Customizable Output**
  - Control summary length
  - Adjust number of sentences (extractive methods)
  - Save summaries to file

- **Production Ready**
  - Error handling and validation
  - Progress indicators
  - Clean command-line interface

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/text-summarization-tool.git
   cd text-summarization-tool
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python summarizer.py --help
   ```

## Usage

### Basic Usage

**Summarize a text file:**
```bash
python summarizer.py -i document.txt
```

**Direct text input:**
```bash
python summarizer.py -t "Your long text goes here..."
```

### Advanced Usage

**Specify summarization method:**
```bash
# Abstractive summarization (default)
python summarizer.py -i document.txt -m abstractive

# LSA extractive summarization
python summarizer.py -i document.txt -m lsa

# Luhn extractive summarization
python summarizer.py -i document.txt -m luhn

# TextRank extractive summarization
python summarizer.py -i document.txt -m textrank
```

**Control summary length:**
```bash
# For extractive methods (number of sentences)
python summarizer.py -i document.txt -m lsa -s 5

# For abstractive method (token length)
python summarizer.py -i document.txt -m abstractive --max-length 200 --min-length 100
```

**Save output to file:**
```bash
python summarizer.py -i document.txt -o summary.txt
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input file path | - |
| `-t, --text` | Direct text input | - |
| `-o, --output` | Output file path | - |
| `-m, --method` | Summarization method (abstractive, lsa, luhn, textrank) | abstractive |
| `-s, --sentences` | Number of sentences (extractive methods) | 3 |
| `--max-length` | Maximum length (abstractive method) | 150 |
| `--min-length` | Minimum length (abstractive method) | 50 |

## Examples

### Example 1: Quick Summary
```bash
python summarizer.py -i research_paper.txt -s 5
```

### Example 2: Detailed Abstractive Summary
```bash
python summarizer.py -i article.txt -m abstractive --max-length 300 --min-length 150 -o summary.txt
```

### Example 3: Compare Methods
```bash
# Try different methods on the same document
python summarizer.py -i document.txt -m lsa -o lsa_summary.txt
python summarizer.py -i document.txt -m textrank -o textrank_summary.txt
python summarizer.py -i document.txt -m abstractive -o abstractive_summary.txt
```

## How It Works

### Abstractive Summarization
Uses Facebook's BART (Bidirectional and Auto-Regressive Transformers) model to generate new sentences that capture the essence of the original text. This method can produce more natural-sounding summaries but requires more computational resources.

### Extractive Summarization
Selects the most important sentences from the original text:
- **LSA**: Uses singular value decomposition to identify key concepts
- **Luhn**: Ranks sentences based on word frequency and significance
- **TextRank**: Applies Google's PageRank algorithm to sentences

## Libraries and Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| `transformers` | Abstractive summarization with BART | 4.35.0 |
| `torch` | Deep learning framework | 2.1.0 |
| `nltk` | Natural language processing | 3.8.1 |
| `sumy` | Extractive summarization algorithms | 0.11.0 |
| `numpy` | Numerical operations | 1.24.3 |
| `sentencepiece` | Tokenization | 0.1.99 |

## Project Structure

```
text-summarization-tool/
│
├── summarizer.py          # Main application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
│
└── examples/             # Example documents (optional)
    ├── sample1.txt
    └── sample2.txt
```

## Performance Considerations

- **Abstractive summarization**: First run downloads ~1.6GB model (BART)
- **Memory usage**: Abstractive method requires ~2-4GB RAM
- **Speed**: Extractive methods are faster; abstractive provides better quality
- **Text length**: Long documents (>1000 words) are automatically chunked

## Troubleshooting

**Issue: Model download fails**
```bash
# Manually download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

**Issue: Out of memory**
```bash
# Use extractive methods for large documents
python summarizer.py -i large_file.txt -m lsa
```

**Issue: Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Facebook AI Research for the BART model
- Hugging Face for the transformers library
- The SUMY project for extractive summarization algorithms

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for the NLP community**