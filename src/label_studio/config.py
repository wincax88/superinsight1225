"""
Label Studio configuration and integration setup
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LabelStudioProject:
    """Label Studio project configuration"""
    id: Optional[int] = None
    title: str = "SuperInsight Annotation Project"
    description: str = "AI data annotation project"
    label_config: str = ""
    expert_instruction: str = ""
    show_instruction: bool = True
    show_skip_button: bool = True
    enable_empty_annotation: bool = False
    show_annotation_history: bool = True
    organization: Optional[int] = None
    color: str = "#FFFFFF"
    maximum_annotations: int = 1
    is_published: bool = False
    model_version: str = ""
    is_draft: bool = True
    created_by: Optional[int] = None
    created_at: Optional[str] = None
    min_annotations_to_start_training: int = 10
    start_training_on_annotation_update: bool = False
    show_collab_predictions: bool = True
    num_tasks_with_annotations: int = 0
    task_number: int = 0
    useful_annotation_number: int = 0
    ground_truth_number: int = 0
    skipped_annotations_number: int = 0
    total_annotations_number: int = 0
    total_predictions_number: int = 0
    sampling: str = "Sequential sampling"
    show_ground_truth_first: bool = False
    show_overlap_first: bool = False
    overlap_cohort_percentage: int = 100
    task_data_login: Optional[str] = None
    task_data_password: Optional[str] = None
    control_weights: Dict[str, float] = None
    parsed_label_config: Dict[str, Any] = None
    evaluate_predictions_automatically: bool = False
    config_has_control_tags: bool = True
    skip_queue: str = "REQUEUE_FOR_OTHERS"
    reveal_preannotations_interactively: bool = False
    pinned_at: Optional[str] = None
    finished_task_number: int = 0
    queue_total: int = 0
    queue_done: int = 0


class LabelStudioConfig:
    """Label Studio configuration manager"""
    
    def __init__(self):
        self.base_url = settings.label_studio.label_studio_url
        self.api_token = settings.label_studio.label_studio_api_token
        self.project_id = settings.label_studio.label_studio_project_id
    
    def get_default_label_config(self, annotation_type: str = "text_classification") -> str:
        """Get default label configuration for different annotation types"""
        
        configs = {
            "text_classification": """
            <View>
              <Text name="text" value="$text"/>
              <Choices name="sentiment" toName="text">
                <Choice value="positive"/>
                <Choice value="negative"/>
                <Choice value="neutral"/>
              </Choices>
            </View>
            """,
            
            "named_entity_recognition": """
            <View>
              <Text name="text" value="$text"/>
              <Labels name="label" toName="text">
                <Label value="PERSON" background="red"/>
                <Label value="ORG" background="blue"/>
                <Label value="LOC" background="green"/>
                <Label value="MISC" background="yellow"/>
              </Labels>
            </View>
            """,
            
            "text_generation": """
            <View>
              <Text name="input_text" value="$input_text"/>
              <TextArea name="generated_text" toName="input_text" 
                       placeholder="Enter generated text here..." 
                       rows="5" maxSubmissions="1"/>
            </View>
            """,
            
            "question_answering": """
            <View>
              <Text name="context" value="$context"/>
              <Text name="question" value="$question"/>
              <TextArea name="answer" toName="context" 
                       placeholder="Enter answer here..." 
                       rows="3" maxSubmissions="1"/>
            </View>
            """,
            
            "image_classification": """
            <View>
              <Image name="image" value="$image"/>
              <Choices name="class" toName="image">
                <Choice value="cat"/>
                <Choice value="dog"/>
                <Choice value="other"/>
              </Choices>
            </View>
            """,
            
            "object_detection": """
            <View>
              <Image name="image" value="$image"/>
              <RectangleLabels name="label" toName="image">
                <Label value="Person" background="red"/>
                <Label value="Car" background="blue"/>
                <Label value="Bike" background="green"/>
              </RectangleLabels>
            </View>
            """
        }
        
        return configs.get(annotation_type, configs["text_classification"])
    
    def get_project_config(self, 
                          title: str = "SuperInsight Project",
                          description: str = "AI Data Annotation Project",
                          annotation_type: str = "text_classification") -> Dict[str, Any]:
        """Get complete project configuration"""
        
        return {
            "title": title,
            "description": description,
            "label_config": self.get_default_label_config(annotation_type),
            "expert_instruction": "Please annotate the data according to the guidelines.",
            "show_instruction": True,
            "show_skip_button": True,
            "enable_empty_annotation": False,
            "show_annotation_history": True,
            "color": "#1f77b4",
            "maximum_annotations": 1,
            "is_published": False,
            "is_draft": False,
            "min_annotations_to_start_training": 10,
            "start_training_on_annotation_update": False,
            "show_collab_predictions": True,
            "sampling": "Sequential sampling",
            "show_ground_truth_first": False,
            "show_overlap_first": False,
            "overlap_cohort_percentage": 100,
            "evaluate_predictions_automatically": False,
            "skip_queue": "REQUEUE_FOR_OTHERS",
            "reveal_preannotations_interactively": True
        }
    
    def get_webhook_config(self, webhook_url: str) -> Dict[str, Any]:
        """Get webhook configuration for quality checks"""
        
        return {
            "url": webhook_url,
            "send_payload": True,
            "send_for_all_actions": False,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_token}"
            },
            "actions": [
                "ANNOTATION_CREATED",
                "ANNOTATION_UPDATED", 
                "ANNOTATION_DELETED",
                "TASK_COMPLETED"
            ]
        }
    
    def get_ml_backend_config(self, ml_backend_url: str) -> Dict[str, Any]:
        """Get ML backend configuration for AI predictions"""
        
        return {
            "url": ml_backend_url,
            "title": "SuperInsight AI Backend",
            "description": "AI prediction service for pre-annotation",
            "model_version": "1.0.0",
            "is_interactive": True,
            "timeout": 30.0,
            "extra_params": {
                "confidence_threshold": 0.5,
                "max_predictions": 10
            }
        }
    
    def validate_config(self) -> bool:
        """Validate Label Studio configuration"""
        
        if not self.base_url:
            logger.error("Label Studio URL is not configured")
            return False
        
        if not self.api_token:
            logger.warning("Label Studio API token is not configured")
            # API token might not be required for local development
        
        return True


# Global Label Studio configuration instance
label_studio_config = LabelStudioConfig()