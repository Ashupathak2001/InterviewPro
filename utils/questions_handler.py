import json
import random
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Enums for categories and subcategories
class QuestionCategory:
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    HR_GENERAL = "hr_general"

class TechnicalField:
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_STRUCTURES_ALGORITHMS = "data_structures_algorithms"
    DATABASE_SYSTEMS = "database_systems"
    WEB_DEVELOPMENT = "web_development"
    MOBILE_DEVELOPMENT = "mobile_development"
    MACHINE_LEARNING = "machine_learning"
    CYBERSECURITY = "cybersecurity"
    CLOUD_COMPUTING = "cloud_computing"
    DEVOPS = "devops"

class BehavioralSubcategory:
    TEAMWORK_COLLABORATION = "teamwork_collaboration"
    COMMUNICATION = "communication"
    PROBLEM_SOLVING = "problem_solving"

class HrGeneralSubcategory:
    BASIC_BACKGROUND = "basic_background"
    MOTIVATION_GOALS = "motivation_goals"


@dataclass
class Question:
    id: str
    text: str
    category: str
    subcategory: Optional[str] = None
    field: Optional[str] = None
    tags: List[str] = None


class QuestionsHandler:
    def __init__(self, json_file_path: str = "questions.json"):
        """
        Initialize the questions handler with the JSON file.
        Args:
            json_file_path: Path to the questions.json file
        """
        self.json_file_path = json_file_path
        self.questions_data = self._load_questions()
        self.all_questions = self._parse_questions()

    def get_all_fields(self) -> List[str]:
        """Get all unique technical fields/categories."""
        fields = set()
        for question in self.all_questions:
            if question.field:
                fields.add(question.field)
        return sorted(list(fields))

    def _load_questions(self) -> Dict:
        """Load questions from JSON file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Questions file not found: {self.json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.json_file_path}")

    def _parse_questions(self) -> List[Question]:
        """Parse questions from JSON data into Question objects."""
        questions = []
        question_id = 1

        interview_questions = self.questions_data.get("interview_questions", {})

        # Process technical questions
        technical_data = interview_questions.get("technical", {})
        for field, question_list in technical_data.items():
            if isinstance(question_list, list):
                for question_text in question_list:
                    questions.append(Question(
                        id=f"technical_{field}_{question_id}",
                        text=question_text,
                        category=QuestionCategory.TECHNICAL,
                        subcategory=field,
                        field=field,
                        tags=["technical", field]
                    ))
                    question_id += 1
            elif isinstance(question_list, dict):
                # Handle nested structures like frontend/backend under web_development
                for sub_field, sub_questions in question_list.items():
                    if isinstance(sub_questions, list):
                        for question_text in sub_questions:
                            questions.append(Question(
                                id=f"technical_{field}_{sub_field}_{question_id}",
                                text=question_text,
                                category=QuestionCategory.TECHNICAL,
                                subcategory=f"{field}_{sub_field}",
                                field=field,
                                tags=["technical", field, sub_field]
                            ))
                            question_id += 1

        # Process behavioral questions
        behavioral_data = interview_questions.get("behavioral", {})
        for subcategory_key, question_list in behavioral_data.items():
            if isinstance(question_list, list):
                for question_text in question_list:
                    questions.append(Question(
                        id=f"behavioral_{subcategory_key}_{question_id}",
                        text=question_text,
                        category=QuestionCategory.BEHAVIORAL,
                        subcategory=subcategory_key,
                        field="behavioral",
                        tags=["behavioral", subcategory_key]
                    ))
                    question_id += 1

        # Process HR general questions
        hr_data = interview_questions.get("hr_general", {})
        for subcategory_key, question_list in hr_data.items():
            if isinstance(question_list, list):
                for question_text in question_list:
                    questions.append(Question(
                        id=f"hr_general_{subcategory_key}_{question_id}",
                        text=question_text,
                        category=QuestionCategory.HR_GENERAL,
                        subcategory=subcategory_key,
                        field="hr",
                        tags=["hr", subcategory_key]
                    ))
                    question_id += 1

        return questions

    def get_all_questions(self) -> List[Question]:
        """Get all questions."""
        return self.all_questions

    def get_questions_by_category(self, category: str) -> List[Question]:
        """Get questions by category."""
        return [q for q in self.all_questions if q.category == category]

    def get_questions_by_field(self, field: str) -> List[Question]:
        """Get questions by technical field."""
        return [q for q in self.all_questions if q.field == field]

    def get_questions_by_subcategory(self, subcategory: str) -> List[Question]:
        """Get questions by subcategory."""
        return [q for q in self.all_questions if q.subcategory == subcategory]

    def get_random_questions(self, count: int = 10, **filters) -> List[Question]:
        """
        Get random questions with optional filters.
        Args:
            count: Number of questions to return
            **filters: Optional filters (category, field, subcategory)
        """
        filtered_questions = self.all_questions

        # Apply filters
        if 'category' in filters:
            category = filters['category']
            filtered_questions = [q for q in filtered_questions if q.category == category]

        if 'field' in filters:
            field = filters['field']
            filtered_questions = [q for q in filtered_questions if q.field == field]

        if 'subcategory' in filters:
            subcategory = filters['subcategory']
            filtered_questions = [q for q in filtered_questions if q.subcategory == subcategory]

        valid_questions = [q for q in filtered_questions if q is not None]
        return random.sample(valid_questions, min(count, len(valid_questions)))

    def search_questions(self, query: str) -> List[Question]:
        """Search questions by text content."""
        query_lower = query.lower()
        return [q for q in self.all_questions if query_lower in q.text.lower()]

    def get_question_stats(self) -> Dict:
        """Get statistics about the questions database."""
        stats = {
            'total_questions': len(self.all_questions),
            'by_category': {},
            'by_field': {},
            'by_subcategory': {}
        }
        for question in self.all_questions:
            # Count by category
            cat_name = question.category
            stats['by_category'][cat_name] = stats['by_category'].get(cat_name, 0) + 1

            # Count by field
            if question.field:
                stats['by_field'][question.field] = stats['by_field'].get(question.field, 0) + 1

            # Count by subcategory
            if question.subcategory:
                stats['by_subcategory'][question.subcategory] = stats['by_subcategory'].get(question.subcategory, 0) + 1

        return stats

    def create_interview_set(self,
                           total_questions: int = 20,
                           technical_ratio: float = 0.6,
                           behavioral_ratio: float = 0.3,
                           hr_ratio: float = 0.1,
                           include_web_dev: bool = True,
                           include_mobile_dev: bool = True) -> List[Question]:
        """
        Create a balanced interview question set.
        Args:
            total_questions: Total number of questions
            technical_ratio: Ratio of technical questions (0.0 to 1.0)
            behavioral_ratio: Ratio of behavioral questions (0.0 to 1.0)
            hr_ratio: Ratio of HR questions (0.0 to 1.0)
            include_web_dev: Whether to include web development questions
            include_mobile_dev: Whether to include mobile development questions
        """
        # Calculate question counts
        technical_count = max(0, min(int(total_questions * technical_ratio), total_questions))
        behavioral_count = max(0, min(int(total_questions * behavioral_ratio), total_questions - technical_count))
        hr_count = max(0, min(int(total_questions * hr_ratio), total_questions - technical_count - behavioral_count))

        interview_questions = []

        # Add technical questions
        technical_questions = self.get_questions_by_category(QuestionCategory.TECHNICAL)
        if technical_count > 0:
            selected = random.sample(technical_questions, min(technical_count, len(technical_questions)))
            interview_questions.extend(selected)

        # Add behavioral questions
        if behavioral_count > 0:
            behavioral_questions = self.get_questions_by_category(QuestionCategory.BEHAVIORAL)
            selected = random.sample(behavioral_questions, min(behavioral_count, len(behavioral_questions)))
            interview_questions.extend(selected)

        # Add HR questions
        if hr_count > 0:
            hr_questions = self.get_questions_by_category(QuestionCategory.HR_GENERAL)
            selected = random.sample(hr_questions, min(hr_count, len(hr_questions)))
            interview_questions.extend(selected)

        # Shuffle the final set
        random.shuffle(interview_questions)

        return interview_questions[:total_questions]

    def export_questions_to_csv(self, filename: str, questions: List[Question] = None):
        """Export questions to CSV file."""
        import csv
        if questions is None:
            questions = self.all_questions
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'text', 'category', 'subcategory', 'field', 'tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for question in questions:
                writer.writerow({
                    'id': question.id,
                    'text': question.text,
                    'category': question.category,
                    'subcategory': question.subcategory or '',
                    'field': question.field or '',
                    'tags': ','.join(question.tags) if question.tags else ''
                })