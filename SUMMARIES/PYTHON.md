# Python Project Structure

## data/papers/2405.21060/source/structure/code.py
```python
def segsum(x)
    """
    Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
    which is equivalent to a scalar SSM.
    """

def ssd(X, A, B, C, block_len, initial_states)
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """

```

## data/papers/2406.01981/source/figures/plots.py
```python
def bar_chat_1_4B_pythia_vs_zyda()

def ablations_410m_barchart()

def pythia_zyda_scaling_plot()

def ablations_plot()

def deduplication_ablations()

```

## data/papers/2407.17465/source/arXiv/code/umup_extension_hardtanh.py
```python
def hardtanh(x, constraint)

```

## src/scripts/arxiv_client.py
```python
class ArxivClient
    """Client for interacting with arXiv API and downloading papers."""

    def __init__(self, papers_dir: str | Path)
        """
        Initialize ArxivClient.
        Args:
            papers_dir: Base directory for paper storage
        """

    def _wait_for_rate_limit(self)
        """Enforce rate limiting between requests."""

    def get_paper_dir(self, arxiv_id: str) -> Path
        """Get paper's directory, creating if needed."""

    def get_paper_status(self, arxiv_id: str) -> dict
        """
        Get current status of paper downloads.
        Returns:
            dict with keys:
                - has_pdf: Whether PDF exists
                - has_source: Whether source exists
                - pdf_size: Size of PDF if it exists
                - source_size: Size of source directory if it exists
        """

    def fetch_metadata(self, arxiv_id: str) -> Paper
        """
        Fetch paper metadata from arXiv API.
        Args:
            arxiv_id: The arXiv identifier
        Returns:
            Paper: Constructed Paper object
        Raises:
            ValueError: If API response is invalid
            Exception: For network or parsing errors
        """

    def _parse_arxiv_response(self, xml_text: str, arxiv_id: str) -> Paper
        """Parse ArXiv API response XML into Paper object."""

    def get_pdf_url(self, arxiv_id: str) -> str
        """Get PDF URL from arXiv ID."""

    def get_source_url(self, arxiv_id: str) -> str
        """Get source URL from arXiv ID."""

    def download_pdf(self, arxiv_id: str) -> bool
        """
        Download PDF for a paper.
        Args:
            arxiv_id: Paper ID to download
        Returns:
            bool: True if successful
        """

    def download_source(self, arxiv_id: str) -> bool
        """
        Download and extract source files for a paper.
        Args:
            arxiv_id: Paper ID to download
        Returns:
            bool: True if successful
        """

    def download_paper(self, arxiv_id: str, skip_existing: bool) -> bool
        """
        Download both PDF and source files for a paper.
        Args:
            arxiv_id: Paper ID to download
            skip_existing: Skip downloads if files exist
        Returns:
            bool: True if all downloads successful
        """


def is_within_directory(directory, target)

def safe_extract(tar, path, members)

```

## src/scripts/asset_manager.py
```python
class PaperAssetManager
    """Manages paper assets including PDFs, source files, and markdown conversions."""

    def __init__(self, papers_dir: str | Path, arxiv_client: Optional[ArxivClient], markdown_service: Optional[MarkdownService])

    def find_missing_pdfs(self) -> list[str]
        """Find papers missing PDF downloads."""

    def find_missing_source(self) -> list[str]
        """Find papers missing source files."""

    def find_pending_markdown(self) -> list[str]
        """Find papers with source but no markdown."""

    def download_pdfs(self, force: bool) -> dict[[str, bool]]
        """Download PDFs for papers missing them."""

    def download_source(self, force: bool) -> dict[[str, bool]]
        """Download source files for papers missing them."""

    def convert_markdown(self, force: bool) -> dict[[str, bool]]
        """Convert papers with source to markdown."""

    def ensure_all_assets(self, force: bool, retry_failed: bool)
        """Ensure all papers have complete assets."""


def main()
    """Command-line interface."""

```

## src/scripts/frontend/generate_html.py
```python
def format_authors(authors: str | list[str]) -> str
    """Format author list consistently."""

def normalize_datetime(date_str: str | None) -> datetime | None
    """Parse datetime string to UTC datetime and strip timezone info."""

def get_last_visited(paper: Dict[[str, Any]]) -> str
    """Compute the most recent interaction time for a paper."""

def preprocess_paper(paper: Dict[[str, Any]]) -> Dict[[str, Any]]
    """Process a single paper entry."""

def preprocess_papers(papers: Dict[[str, Any]]) -> Dict[[str, Any]]
    """Process all papers and prepare them for display."""

def generate_html(data_path: str, template_path: str, output_path: str) -> None
    """
    Generate HTML page from papers data and template.
    Args:
        data_path: Path to papers YAML file
        template_path: Path to HTML template file
        output_path: Path where generated HTML should be written
    """

```

## src/scripts/github_client.py
```python
def patch_schema_change(issue)

class GithubClient
    """Handles GitHub API interactions."""

    def __init__(self, token: str, repo: str)

    def get_open_issues(self) -> List[Dict[[str, Any]]]
        """Fetch open issues with paper or reading-session labels."""

    def close_issue(self, issue_number: int) -> bool
        """Close an issue with comment."""


```

## src/scripts/markdown_service.py
```python
class MarkdownService
    """Manages the conversion of LaTeX papers to Markdown format."""

    def __init__(self, papers_dir: str | Path)
        """
        Initialize MarkdownService.
        Args:
            papers_dir: Base directory for paper storage
        """

    def _load_failed_conversions(self)
        """Load record of failed conversions with timestamps."""

    def _save_failed_conversions(self)
        """Save record of failed conversions."""

    def _record_failure(self, arxiv_id: str, error: str)
        """Record a conversion failure with timestamp."""

    def _clear_failure(self, arxiv_id: str)
        """Clear a failure record after successful conversion."""

    def should_retry_conversion(self, arxiv_id: str, retry_after_hours: int) -> bool
        """
        Check if we should retry a failed conversion.
        Args:
            arxiv_id: Paper ID to check
            retry_after_hours: Hours to wait before retrying
        Returns:
            bool: True if enough time has passed to retry
        """

    def convert_paper(self, arxiv_id: str, force: bool, tex_file: Optional[Path]) -> bool
        """
        Convert a paper's LaTeX source to Markdown.
        Args:
            arxiv_id: Paper ID to convert
            force: Force conversion even if previously failed
            tex_file: Optional specific tex file to use for conversion
        """

    def retry_failed_conversions(self, force: bool)
        """
        Retry converting papers that previously failed.
        Args:
            force: Force retry all failed conversions regardless of timing
        """

    def get_conversion_status(self, arxiv_id: str) -> dict
        """
        Get the current conversion status for a paper.
        Args:
            arxiv_id: Paper ID to check
        Returns:
            dict: Status information including:
                - has_markdown: Whether markdown exists
                - has_source: Whether source exists
                - failed: Whether conversion previously failed
                - last_attempt: Timestamp of last attempt if failed
                - error: Error message if failed
        """


```

## src/scripts/models.py
```python
class Paper(BaseModel)
    """Schema for paper metadata"""

class ReadingSession(BaseModel)
    """Schema for reading session events"""

class PaperVisitEvent(BaseModel)
    """Schema for paper visit events"""

class Config

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
        Convert a LaTeX file to Markdown using Pandoc.
        Args:
            tex_file: Path to LaTeX file
            output_file: Optional output path, defaults to same name with .md extension
        Returns:
            bool: True if conversion successful
        """


def create_default_config(paper_dir: Path) -> PandocConfig
    """Create default Pandoc configuration for a paper directory."""

```

## src/scripts/paper_manager.py
```python
class PaperManager
    """Manages paper metadata and event storage."""

    def __init__(self, data_dir: Path, arxiv_client: Optional[ArxivClient])
        """Initialize PaperManager with data directory and optional ArxivClient."""

    def _needs_hydration(self, paper: Paper) -> bool
        """Check if paper needs metadata hydration."""

    def _hydrate_metadata(self, paper: Paper) -> Paper
        """Fetch missing metadata from arXiv API."""

    def get_paper(self, arxiv_id: str) -> Paper
        """Get paper metadata, hydrating if necessary."""

    def fetch_new_paper(self, arxiv_id: str) -> Paper
        """Fetch paper metadata from ArXiv."""

    def get_or_create_paper(self, arxiv_id: str) -> Paper
        """Get existing paper or create new one."""

    def create_paper(self, paper: Paper) -> None
        """Create new paper directory and initialize metadata."""

    def save_metadata(self, paper: Paper) -> None
        """Save paper metadata to file."""

    def load_metadata(self, arxiv_id: str) -> Paper
        """Load paper metadata from file."""

    def append_event(self, arxiv_id: str, event: PaperVisitEvent | ReadingSession) -> None
        """Append event to paper's event log."""

    def update_reading_time(self, arxiv_id: str, duration_seconds: int) -> None
        """Update paper's total reading time."""

    def get_modified_files(self) -> set[str]
        """Get set of modified file paths."""

    def clear_modified_files(self) -> None
        """Clear set of modified files."""

    def update_main_tex_file(self, arxiv_id: str, tex_file: Path) -> None
        """Update paper's main TeX file path."""


```

## src/scripts/process_events.py
```python
class EventProcessor
    """Processes GitHub issues into paper events."""

    def __init__(self, papers_dir: str | Path)
        """Initialize EventProcessor with GitHub credentials and paths."""

    def process_paper_issue(self, issue_data: Dict[[str, Any]]) -> bool
        """Process paper registration issue."""

    def process_reading_issue(self, issue_data: Dict[[str, Any]]) -> bool
        """Process reading session issue."""

    def update_registry(self) -> None
        """Update central registry with modified papers."""

    def process_all_issues(self) -> None
        """Process all open issues."""


def main()
    """Main entry point for processing paper events."""

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

## tests/conftest.py
```python
def mock_pandoc()
    """Mock pandoc for all tests."""

def test_dir(tmp_path)
    """Create a clean test directory."""

def paper_dir(test_dir)
    """Create a test paper directory."""

def sample_paper()
    """Create sample Paper object."""

def source_dir(paper_dir)
    """Create source directory with test TeX content."""

def mock_pandoc_run(cmd, capture_output, cwd, text)

```

## tests/test_arxiv_client.py
```python
def client(test_dir)
    """Create ArxivClient instance with rate limiting disabled."""

def arxiv_success_response()
    """Sample successful arXiv API response."""

class TestArxivClient

    def test_get_paper_dir(self, client)
        """Test paper directory creation."""

    def test_get_paper_status_empty(self, client)
        """Test paper status for paper with no files."""

    def test_get_paper_status_with_files(self, client)
        """Test paper status with existing files."""

    def test_fetch_metadata_success(self, client, arxiv_success_response)
        """Test successful metadata fetch with extended fields."""

    def test_fetch_metadata_api_error(self, client)
        """Test handling of API error responses."""

    def test_fetch_metadata_invalid_xml(self, client)
        """Test handling of invalid XML responses."""

    def test_download_pdf_success(self, client)
        """Test successful PDF download."""

    def test_download_pdf_failure(self, client)
        """Test handling of PDF download failures."""

    def test_download_source_failure(self, client)
        """Test handling of source download failures."""

    def test_download_paper_complete(self, client)
        """Test downloading complete paper with PDF and source."""

    def test_rate_limiting(self, client)
        """Test rate limiting between requests."""


def test_download_source_success(self, client)
    """Test successful source download."""

```

## tests/test_asset_manager.py
```python
def manager(test_dir)
    """Create AssetManager with mocked dependencies."""

def test_ensure_all_assets(manager)
    """Test processing all papers."""

def test_convert_markdown(manager)
    """Test converting papers to markdown."""

def get_paper_status(arxiv_id)

def get_conversion_status(arxiv_id)

```

## tests/test_github_client.py
```python
def client()
    """Create GithubClient instance."""

class TestGithubClient

    def test_get_open_issues(self, client)
        """Test fetching open issues."""

    def test_get_open_issues_error(self, client)
        """Test handling API errors in issue fetching."""

    def test_close_issue_success(self, client)
        """Test successful issue closing."""

    def test_close_issue_comment_error(self, client)
        """Test handling comment creation error."""

    def test_close_issue_close_error(self, client)
        """Test handling issue closing error."""


```

## tests/test_markdown_service.py
```python
def paper_manager(test_dir)
    """Create PaperManager instance."""

def service(test_dir)
    """Create MarkdownService instance."""

def setup_metadata_with_tex(paper_dir)
    """Setup metadata.json with main_tex_file specified."""

class TestMarkdownService

    def test_convert_with_metadata_tex_file(self, service, source_dir, setup_metadata_with_tex, mock_pandoc)
        """Test conversion using main_tex_file from metadata."""

    def test_convert_with_invalid_metadata_tex_file(self, service, source_dir, setup_metadata_with_tex)
        """Test fallback when metadata specifies non-existent tex file."""

    def test_convert_with_paper_manager_update(self, service, source_dir, paper_manager, mock_pandoc)
        """Test conversion after updating main_tex_file via PaperManager."""

    def test_convert_paper_success(self, service, source_dir, mock_pandoc)
        """Test successful paper conversion."""

    def test_convert_paper_no_source(self, service, paper_dir)
        """Test conversion without source files."""

    def test_force_reconversion(self, service, source_dir, mock_pandoc)
        """Test forced reconversion."""

    def test_skip_recent_failure(self, service, paper_dir)
        """Test that recent failures are skipped."""


```

## tests/test_pandoc_utils.py
```python
def mock_subprocess_run()
    """Mock successful subprocess run."""

def test_tex_content()
    """Sample LaTeX content for testing."""

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

def mock_success()

```

## tests/test_paper_manager.py
```python
def manager(test_dir)
    """Create PaperManager instance with test directory."""

class TestPaperManager

    def test_get_paper_not_found(self, manager)
        """Test getting non-existent paper."""

    def test_create_and_get_paper(self, manager, sample_paper)
        """Test creating and retrieving a paper."""

    def test_get_or_create_paper_existing(self, manager, sample_paper)
        """Test get_or_create with existing paper."""

    def test_get_or_create_paper_new(self, manager)
        """Test get_or_create fetches new paper."""

    def test_update_reading_time(self, manager, sample_paper)
        """Test updating paper reading time."""

    def test_append_event(self, manager, sample_paper)
        """Test appending reading session event."""

    def test_modified_files_tracking(self, manager, sample_paper)
        """Test tracking of modified files."""

    def test_save_load_metadata(self, manager, sample_paper)
        """Test metadata serialization."""

    def test_concurrent_event_writing(self, manager, sample_paper)
        """Test concurrent writing of multiple events."""


```

## tests/test_paper_manager_hydration.py
```python
def manager(test_dir)
    """Create PaperManager instance with test directory."""

def paper_with_missing_fields(sample_paper)
    """Create paper missing optional metadata fields."""

def complete_paper(sample_paper)
    """Create paper with all metadata fields."""

class TestPaperManagerHydration

    def test_needs_hydration_missing_fields(self, manager, paper_with_missing_fields)
        """Test hydration check with missing fields."""

    def test_needs_hydration_complete(self, manager, complete_paper)
        """Test hydration check with complete metadata."""

    def test_needs_hydration_empty_tags(self, manager, complete_paper)
        """Test hydration check with empty tags list."""

    def test_hydrate_metadata_success(self, manager, paper_with_missing_fields, complete_paper)
        """Test successful metadata hydration."""

    def test_hydrate_metadata_failure(self, manager, paper_with_missing_fields)
        """Test handling of hydration failure."""

    def test_get_paper_triggers_hydration(self, manager, paper_with_missing_fields)
        """Test that get_paper initiates hydration when needed."""

    def test_create_paper_with_hydration(self, manager, paper_with_missing_fields, complete_paper)
        """Test that create_paper performs hydration."""


```

## tests/test_process_events.py
```python
def sample_paper_issue(sample_paper)
    """Create sample paper registration issue."""

def event_processor(tmp_path)
    """Create EventProcessor with temp directory."""

class TestEventProcessor

    def test_process_paper_issue(self, event_processor, sample_paper_issue, sample_paper)
        """Test processing paper registration issue."""

    def test_process_reading_issue(self, event_processor, sample_paper)
        """Test processing reading session issue."""

    def test_process_reading_issue_invalid_data(self, event_processor)
        """Test processing invalid reading session data."""

    def test_update_registry(self, event_processor, sample_paper, tmp_path)
        """Test updating registry file."""

    def test_process_all_issues(self, event_processor, sample_paper_issue)
        """Test processing multiple issue types."""

    def test_process_no_issues(self, event_processor)
        """Test behavior when no issues exist."""

    def test_github_api_error(self, event_processor)
        """Test handling of GitHub API errors."""


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
