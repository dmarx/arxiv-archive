import pytest
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from scripts.process_events import EventProcessor, Paper, ReadingSession, PaperRegistrationEvent

@pytest.fixture
def sample_paper_issue():
    """Fixture for a sample paper registration issue"""
    return {
        "number": 1,
        "html_url": "https://github.com/user/repo/issues/1",
        "state": "open",
        "created_at": "2024-01-01T00:00:00Z",
        "labels": [{"name": "paper"}],
        "body": json.dumps({
            "arxivId": "2401.00001",
            "title": "Test Paper",
            "authors": "Test Author",
            "abstract": "Test abstract",
            "url": "https://arxiv.org/abs/2401.00001",
            "timestamp": "2024-01-01T00:00:00Z",
            "rating": "novote"
        })
    }

@pytest.fixture
def sample_reading_session_issue():
    """Fixture for a sample reading session issue"""
    return {
        "number": 2,
        "html_url": "https://github.com/user/repo/issues/2",
        "state": "open",
        "created_at": "2024-01-01T01:00:00Z",
        "labels": [{"name": "reading-session"}],
        "body": json.dumps({
            "type": "reading_session",
            "arxivId": "2401.00001",
            "timestamp": "2024-01-01T01:00:00Z",
            "duration_minutes": 30,
            "paper_url": "https://arxiv.org/abs/2401.00001",
            "paper_title": "Test Paper"
        })
    }

@pytest.fixture
def event_processor(tmp_path):
    """Fixture for EventProcessor with temporary directory"""
    with patch.dict('os.environ', {
        'GITHUB_TOKEN': 'fake_token',
        'GITHUB_REPOSITORY': 'user/repo'
    }):
        processor = EventProcessor()
        processor.papers_dir = tmp_path / "papers"
        processor.papers_dir.mkdir(parents=True)
        return processor

@pytest.mark.asyncio
async def test_get_open_issues(event_processor):
    """Test fetching open issues"""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = [
        {"labels": [{"name": "paper"}]},
        {"labels": [{"name": "reading-session"}]},
        {"labels": [{"name": "other"}]}
    ]
    
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    issues = await event_processor.get_open_issues(mock_session)
    assert len(issues) == 2
    assert all(any(label["name"] in ["paper", "reading-session"] 
              for label in issue["labels"]) for issue in issues)

def test_create_paper_from_issue(event_processor, sample_paper_issue):
    """Test paper creation from issue data"""
    paper_data = json.loads(sample_paper_issue["body"])
    paper = event_processor.create_paper_from_issue(sample_paper_issue, paper_data)
    
    assert isinstance(paper, Paper)
    assert paper.arxiv_id == "2401.00001"
    assert paper.title == "Test Paper"
    assert paper.issue_number == 1
    assert paper.total_reading_time_minutes == 0

def test_ensure_paper_directory(event_processor):
    """Test paper directory creation"""
    arxiv_id = "2401.00001"
    paper_dir = event_processor.ensure_paper_directory(arxiv_id)
    
    assert paper_dir.exists()
    assert paper_dir.is_dir()
    assert paper_dir.name == arxiv_id

def test_save_and_load_paper_metadata(event_processor, sample_paper_issue):
    """Test saving and loading paper metadata"""
    paper_data = json.loads(sample_paper_issue["body"])
    paper = event_processor.create_paper_from_issue(sample_paper_issue, paper_data)
    
    # Save metadata
    event_processor.ensure_paper_directory(paper.arxiv_id)
    event_processor.save_paper_metadata(paper)
    
    # Load metadata
    loaded_paper = event_processor.load_paper_metadata(paper.arxiv_id)
    
    assert loaded_paper is not None
    assert loaded_paper.arxiv_id == paper.arxiv_id
    assert loaded_paper.title == paper.title
    assert loaded_paper.model_dump() == paper.model_dump()

def test_append_event(event_processor):
    """Test appending events to log file"""
    arxiv_id = "2401.00001"
    event = ReadingSession(
        arxivId=arxiv_id,
        timestamp="2024-01-01T00:00:00Z",
        duration_minutes=30,
        issue_url="https://github.com/user/repo/issues/1"
    )
    
    event_processor.ensure_paper_directory(arxiv_id)
    event_processor.append_event(arxiv_id, event)
    
    events_file = event_processor.papers_dir / arxiv_id / "events.log"
    assert events_file.exists()
    
    with events_file.open('r') as f:
        event_data = json.loads(f.readline())
        assert event_data["arxivId"] == arxiv_id
        assert event_data["duration_minutes"] == 30

def test_process_new_paper(event_processor, sample_paper_issue):
    """Test processing a new paper registration"""
    success = event_processor.process_new_paper(sample_paper_issue)
    assert success
    
    arxiv_id = "2401.00001"
    paper = event_processor.load_paper_metadata(arxiv_id)
    assert paper is not None
    assert paper.arxiv_id == arxiv_id
    
    events_file = event_processor.papers_dir / arxiv_id / "events.log"
    with events_file.open('r') as f:
        event_data = json.loads(f.readline())
        assert isinstance(PaperRegistrationEvent.model_validate(event_data), PaperRegistrationEvent)

def test_process_reading_session(event_processor, sample_reading_session_issue):
    """Test processing a reading session"""
    # First create a paper
    event_processor.process_new_paper({
        **sample_reading_session_issue,
        "labels": [{"name": "paper"}],
        "body": json.dumps({
            "arxivId": "2401.00001",
            "title": "Test Paper",
            "authors": "Test Author",
            "abstract": "Test abstract",
            "url": "https://arxiv.org/abs/2401.00001"
        })
    })
    
    # Process reading session
    success = event_processor.process_reading_session(sample_reading_session_issue)
    assert success
    
    # Verify paper was updated
    paper = event_processor.load_paper_metadata("2401.00001")
    assert paper.total_reading_time_minutes == 30
    assert paper.last_read == "2024-01-01T01:00:00Z"

def test_update_registry(event_processor, sample_paper_issue):
    """Test updating the centralized registry"""
    # Process a paper to create some data
    event_processor.process_new_paper(sample_paper_issue)
    
    # Update registry
    event_processor.update_registry()
    
    registry_file = Path("data/papers.yaml")
    assert str(registry_file) in event_processor.modified_files
    
    # Verify registry contents
    with registry_file.open('r') as f:
        registry = yaml.safe_load(f)
        assert "2401.00001" in registry
        assert registry["2401.00001"]["title"] == "Test Paper"

@pytest.mark.asyncio
async def test_close_issues(event_processor):
    """Test closing processed issues"""
    mock_session = AsyncMock()
    
    # Setup mock responses
    mock_comment_response = AsyncMock()
    mock_comment_response.status = 201
    mock_close_response = AsyncMock()
    mock_close_response.status = 200
    
    mock_session.post.return_value.__aenter__.return_value = mock_comment_response
    mock_session.patch.return_value.__aenter__.return_value = mock_close_response
    
    # Add some processed issues
    event_processor.processed_issues = [1, 2]
    
    await event_processor.close_issues(mock_session)
    
    # Verify API calls
    assert mock_session.post.call_count == 2  # One comment per issue
    assert mock_session.patch.call_count == 2  # One close per issue

@pytest.mark.asyncio
async def test_process_all_issues(event_processor, sample_paper_issue, sample_reading_session_issue):
    """Test end-to-end processing of all issues"""
    mock_session = AsyncMock()
    mock_get_response = AsyncMock()
    mock_get_response.status = 200
    mock_get_response.json.return_value = [sample_paper_issue, sample_reading_session_issue]
    
    mock_comment_response = AsyncMock()
    mock_comment_response.status = 201
    mock_close_response = AsyncMock()
    mock_close_response.status = 200
    
    mock_session.get.return_value.__aenter__.return_value = mock_get_response
    mock_session.post.return_value.__aenter__.return_value = mock_comment_response
    mock_session.patch.return_value.__aenter__.return_value = mock_close_response
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        with patch('scripts.process_events.commit_and_push') as mock_commit:
            await event_processor.process_all_issues()
    
    # Verify processing occurred
    paper = event_processor.load_paper_metadata("2401.00001")
    assert paper is not None
    assert paper.total_reading_time_minutes == 30
    
    # Verify commit was attempted
    mock_commit.assert_called_once()
    
    # Verify issues were closed
    assert mock_session.patch.call_count == 2  # Two issues closed
