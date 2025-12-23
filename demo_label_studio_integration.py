#!/usr/bin/env python3
"""
Demo script for Label Studio integration and multi-user collaboration.

This script demonstrates the key features implemented in task 5:
- Label Studio project creation and management
- Multi-user collaboration with role-based access control
- Task assignment and progress tracking
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from uuid import uuid4

# Add src to path for imports
sys.path.append('src')

from label_studio import (
    LabelStudioIntegration,
    ProjectConfig,
    collaboration_manager,
    auth_manager,
    create_demo_users,
    UserRole
)
from models.task import Task
from models.document import Document


async def demo_label_studio_integration():
    """Demonstrate Label Studio integration features"""
    
    print("🚀 SuperInsight Label Studio Integration Demo")
    print("=" * 50)
    
    # 1. Create demo users for collaboration
    print("\n1. Creating demo users...")
    users = create_demo_users()
    
    for user in users:
        print(f"   ✓ Created {user.role.value}: {user.username} ({user.email})")
    
    # 2. Demonstrate authentication
    print("\n2. Testing authentication...")
    admin_user = next(u for u in users if u.role == UserRole.ADMIN)
    
    # Create access token
    token = auth_manager.create_access_token(admin_user.id)
    print(f"   ✓ Generated JWT token for admin (length: {len(token)})")
    
    # Verify token
    current_user = auth_manager.get_current_user(token)
    if current_user:
        print(f"   ✓ Token verified for user: {current_user.username}")
    
    # 3. Create Label Studio project (simulated)
    print("\n3. Creating Label Studio project...")
    
    try:
        integration = LabelStudioIntegration()
        
        project_config = ProjectConfig(
            title="SuperInsight 情感分析项目",
            description="中文文本情感分析标注项目",
            annotation_type="text_classification"
        )
        
        print(f"   ✓ Project configuration created: {project_config.title}")
        print(f"   ✓ Annotation type: {project_config.annotation_type}")
        
        # Note: Actual project creation would require Label Studio server
        print("   ⚠️  Skipping actual project creation (requires Label Studio server)")
        
    except Exception as e:
        print(f"   ⚠️  Label Studio integration: {str(e)}")
    
    # 4. Create sample tasks and documents
    print("\n4. Creating sample tasks...")
    
    sample_documents = [
        Document(
            id=uuid4(),
            source_type="file",
            source_config={"type": "manual", "filename": "sample1.txt"},
            content="这个产品真的很棒，我非常喜欢！",
            metadata={"language": "zh", "domain": "product_review"}
        ),
        Document(
            id=uuid4(),
            source_type="file", 
            source_config={"type": "manual", "filename": "sample2.txt"},
            content="服务态度很差，完全不推荐。",
            metadata={"language": "zh", "domain": "service_review"}
        ),
        Document(
            id=uuid4(),
            source_type="file",
            source_config={"type": "manual", "filename": "sample3.txt"},
            content="价格还可以，质量一般般。",
            metadata={"language": "zh", "domain": "product_review"}
        )
    ]
    
    sample_tasks = []
    for doc in sample_documents:
        task = Task(
            id=uuid4(),
            document_id=doc.id,
            project_id="sentiment_analysis_project",
            ai_predictions=[
                {
                    "model": "sentiment_classifier_v1",
                    "prediction": "positive" if "棒" in doc.content or "喜欢" in doc.content else "negative" if "差" in doc.content else "neutral",
                    "confidence": 0.85,
                    "result": [
                        {
                            "from_name": "sentiment",
                            "to_name": "text",
                            "type": "choices",
                            "value": {
                                "choices": ["positive" if "棒" in doc.content or "喜欢" in doc.content else "negative" if "差" in doc.content else "neutral"]
                            }
                        }
                    ]
                }
            ]
        )
        sample_tasks.append(task)
    
    print(f"   ✓ Created {len(sample_tasks)} sample tasks with AI predictions")
    
    # 5. Demonstrate task assignment
    print("\n5. Demonstrating task assignment...")
    
    # Get annotators
    business_expert = next(u for u in users if u.role == UserRole.BUSINESS_EXPERT)
    tech_expert = next(u for u in users if u.role == UserRole.TECHNICAL_EXPERT)
    annotator = next(u for u in users if u.role == UserRole.OUTSOURCED_ANNOTATOR)
    
    # Assign tasks using different strategies
    task_ids = [task.id for task in sample_tasks]
    user_ids = [business_expert.id, tech_expert.id, annotator.id]
    
    # Round robin assignment
    assignments = collaboration_manager.bulk_assign_tasks(
        task_ids=task_ids,
        user_ids=user_ids,
        assigned_by=admin_user.id,
        strategy="round_robin"
    )
    
    print(f"   ✓ Assigned {len(assignments)} tasks using round-robin strategy")
    
    for assignment in assignments:
        user = collaboration_manager.get_user(assignment.user_id)
        print(f"     - Task {str(assignment.task_id)[:8]}... → {user.username} ({user.role.value})")
    
    # 6. Demonstrate progress tracking
    print("\n6. Generating progress statistics...")
    
    # Simulate some task completion
    for i, assignment in enumerate(assignments[:2]):
        collaboration_manager.update_assignment_status(assignment.id, "completed")
    
    stats = collaboration_manager.get_progress_stats("sentiment_analysis_project")
    
    print(f"   ✓ Total tasks: {stats.total_tasks}")
    print(f"   ✓ Pending tasks: {stats.pending_tasks}")
    print(f"   ✓ Completed tasks: {stats.completed_tasks}")
    print(f"   ✓ Completion rate: {stats.completion_rate:.1%}")
    
    if stats.user_stats:
        print("   ✓ User statistics:")
        for user_id, user_stat in stats.user_stats.items():
            print(f"     - {user_stat['username']}: {user_stat['completed']} completed")
    
    # 7. Demonstrate permission checking
    print("\n7. Testing role-based permissions...")
    
    from label_studio.collaboration import Permission
    
    test_cases = [
        (admin_user, Permission.MANAGE_USERS, True),
        (business_expert, Permission.ANNOTATE, True),
        (annotator, Permission.MANAGE_USERS, False),
        (tech_expert, Permission.EXPORT_DATA, True),
        (annotator, Permission.EXPORT_DATA, False)
    ]
    
    for user, permission, expected in test_cases:
        result = collaboration_manager.check_permission(user.id, permission)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {user.username} ({user.role.value}) - {permission.value}: {result}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey features demonstrated:")
    print("- ✓ Multi-user collaboration with role-based access control")
    print("- ✓ JWT-based authentication and session management")
    print("- ✓ Label Studio project configuration")
    print("- ✓ Task assignment with multiple strategies")
    print("- ✓ Real-time progress tracking")
    print("- ✓ Permission-based access control")
    print("\nNext steps:")
    print("- Start Label Studio server to test full integration")
    print("- Configure webhooks for quality check triggers")
    print("- Set up PostgreSQL database for persistent storage")


if __name__ == "__main__":
    asyncio.run(demo_label_studio_integration())