# Python Project Structure

## src/scripts/download_papers.py
```python
class ArxivDownloader

    def __init__(self, papers_dir: str | Path)
        """
        Initialize the ArXiv paper downloader.
        Args:
            papers_dir: Path to store paper files, default "data/papers"
        """

    def _load_failed_markdown_ids(self)
        """Load the set of arxiv IDs that previously failed markdown conversion."""

    def _add_failed_markdown(self, arxiv_id: str)
        """Add an arxiv ID to the failed markdown tracking file."""

    def get_papers_missing_files(self) -> list[dict]
        """Report current file status for each paper directory."""

    def process_paper(self, session: Any, paper_status: dict) -> bool
        """Process a single paper's downloads and conversions."""

    def convert_to_markdown(self, arxiv_id: str) -> bool
        """Convert LaTeX source to Markdown using enhanced Pandoc conversion."""

    def get_pdf_url(self, arxiv_id: str) -> str
        """Get PDF URL from arXiv ID."""

    def get_source_url(self, arxiv_id: str) -> str
        """Get source URL from arXiv ID."""

    def download_pdf(self, session: Any, arxiv_id: str) -> bool
        """Download PDF for a single paper."""

    def download_source(self, session: Any, arxiv_id: str) -> bool
        """Download and extract source files for a single paper."""

    def download_all_missing(self)
        """Download and process all missing files for papers."""


def download_papers(papers_dir: str | Path)
    """
    CLI entry point for downloading missing paper files.
    Args:
        papers_dir: Path to store paper files, default "data/papers"
    """

def main()
    """CLI entry point using Fire."""

def is_within_directory(directory, target)

def safe_extract(tar, path, members)

```

## src/scripts/pandoc_utils.py
```python
@dataclass
class PandocConfig
    """Configuration for Pandoc conversion."""

class PandocConverter
    """Convert LaTeX papers to Markdown using enhanced Pandoc settings."""

    def __init__(self, config: PandocConfig)
        """Initialize converter with configuration."""

    def _ensure_directories(self)
        """Ensure all required directories exist."""

    def _write_file(self, path: Path, content: str) -> bool
        """Write content to file and verify it exists."""

    def _create_default_files(self)
        """Create default supporting files if not provided."""

    def _verify_files_exist(self) -> bool
        """Verify that all required files exist before running pandoc."""

    def build_pandoc_command(self, input_file: Path, output_file: Path) -> list[str]
        """Build Pandoc command with all necessary arguments."""

    def convert_tex_to_markdown(self, tex_file: Path, output_file: Optional[Path]) -> bool
        """
        Convert a LaTeX file to Markdown using enhanced Pandoc settings.
        Args:
            tex_file: Path to LaTeX file
            output_file: Optional output path, defaults to same name with .md extension
        Returns:
            bool: True if conversion successful
        """


def create_default_config(paper_dir: Path) -> PandocConfig
    """Create default Pandoc configuration for a paper directory."""

```

## src/scripts/process_events.py
```python
class Paper(BaseModel)
    """Schema for paper metadata"""

class ReadingSession(BaseModel)
    """Schema for reading session events"""

class PaperRegistrationEvent(BaseModel)
    """Schema for paper registration events"""

class EventProcessor

    def __init__(self)

    def get_open_issues(self, session) -> list[dict]
        """Fetch all open issues with paper or reading-session labels."""

    def create_paper_from_issue(self, issue_data: dict, paper_data: dict) -> Paper
        """Create a Paper model from issue and paper data."""

    def ensure_paper_directory(self, arxiv_id: str) -> Path
        """Create paper directory if it doesn't exist."""

    def load_paper_metadata(self, arxiv_id: str) -> Paper | None
        """Load paper metadata from file."""

    def save_paper_metadata(self, paper: Paper)
        """Save paper metadata to file."""

    def append_event(self, arxiv_id: str, event: BaseModel)
        """Append an event to the paper's event log."""

    def process_new_paper(self, issue_data: dict) -> bool
        """Process a new paper registration."""

    def process_reading_session(self, issue_data: dict) -> bool
        """Process a reading session event."""

    def close_issues(self, session)
        """Close all successfully processed issues."""

    def update_registry(self)
        """Update the centralized registry file with any modified papers."""

    def process_all_issues(self)
        """Process all open issues."""


def main()
    """Main entry point for processing paper events."""

class Config

```

## src/scripts/tex_utils.py
```python
@dataclass
class TeXFileScore
    """Score details for a TeX file candidate."""

def score_tex_file(tex_file: Path) -> TeXFileScore
    """Score a single TeX file based on ML-focused heuristics."""

def find_main_tex_file(tex_files: Sequence[Path], arxiv_id: str) -> Path | None
    """
    Find the most likely main TeX file from a list of candidates.
    Args:
        tex_files: List of TeX file paths to evaluate
        arxiv_id: ArXiv ID for logging context
    Returns:
        Path to the most likely main TeX file, or None if no valid candidates
    """

```

## tests/test_download_papers.py
```python
def downloader(tmp_path)

def paper_dir(downloader)
    """Create a paper directory for testing."""

def sample_tex_content()

def test_get_papers_missing_files(downloader)

def test_get_urls()

def test_convert_to_markdown(downloader, paper_dir, sample_tex_content)

```

## tests/test_pandoc_utils.py
```python
def mock_subprocess_run()
    """Mock successful subprocess run."""

def test_tex_content()
    """Sample LaTeX content for testing."""

def paper_dir(tmp_path)
    """Create a paper directory with necessary structure."""

def source_dir(paper_dir, test_tex_content)
    """Create source directory with test TeX file."""

def converter(paper_dir)
    """Create PandocConverter instance with test configuration."""

def test_directory_creation(paper_dir, converter)
    """Test that all necessary directories are created."""

def test_supporting_files_creation(paper_dir, converter)
    """Test that all supporting files are created correctly."""

def test_file_verification(paper_dir, converter)
    """Test file verification logic."""

def test_pandoc_command_building(paper_dir, converter)
    """Test pandoc command construction."""

def test_full_conversion_process(paper_dir, source_dir, converter, mock_subprocess_run)
    """Test the complete conversion process."""

def test_real_pandoc_execution(paper_dir, source_dir, converter, test_tex_content)
    """Test with actual pandoc execution."""

def test_error_handling(paper_dir, converter)
    """Test error handling in various scenarios."""

def test_temporary_directory_cleanup(paper_dir, source_dir, converter)
    """Test that temporary directory is properly cleaned up."""

def test_minimal_pandoc_conversion(tmp_path)
    """Test bare minimum pandoc conversion with real pandoc."""

def mock_pandoc_effect()

```

## tests/test_process_events.py
```python
class AsyncContextManagerMock

    def __init__(self, return_value)

    def __aenter__(self)

    def __aexit__(self, exc_type, exc_val, exc_tb)


def mock_response()
    """Create a mock response with common attributes"""

def mock_session(mock_response)
    """Create a mock session with proper context manager methods"""

def sample_paper_issue()
    """Fixture for a sample paper registration issue"""

def sample_reading_session_issue()
    """Fixture for a sample reading session issue"""

def event_processor(tmp_path)
    """Fixture for EventProcessor with temporary directory"""

def test_get_open_issues(event_processor, mock_session)
    """Test fetching open issues"""

def test_create_paper_from_issue(event_processor, sample_paper_issue)
    """Test paper creation from issue data"""

def test_ensure_paper_directory(event_processor)
    """Test paper directory creation"""

def test_save_and_load_paper_metadata(event_processor, sample_paper_issue)
    """Test saving and loading paper metadata"""

def test_append_event(event_processor)
    """Test appending events to log file"""

def test_process_new_paper(event_processor, sample_paper_issue)
    """Test processing a new paper registration"""

def test_process_reading_session(event_processor, sample_reading_session_issue)
    """Test processing a reading session"""

def test_update_registry(event_processor, sample_paper_issue)
    """Test updating the centralized registry"""

def test_close_issues(event_processor, mock_session)
    """Test closing processed issues"""

def test_process_all_issues(event_processor, sample_paper_issue, sample_reading_session_issue)
    """Test end-to-end processing of all issues"""

def make_context_manager(response)

```

## tests/test_tex_utils.py
```python
def tex_dir(tmp_path)
    """Create a temporary directory with test TeX files."""

def create_tex_file(directory: Path, name: str, content: str) -> Path
    """Helper to create a TeX file with given content."""

def test_score_tex_file(tex_dir)

def test_find_main_tex_file_simple(tex_dir)

def test_find_main_tex_file_ml_conference(tex_dir)

def test_find_main_tex_file_empty_list()

def test_score_tex_file_with_inputs(tex_dir)

```
