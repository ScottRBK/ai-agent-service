# Test cases for reranking model evaluation
# Each entry has query, document, and expected relevance (True=relevant, False=irrelevant, None=ambiguous)

RANKING_TEST_CASES = [
    # Technical queries - relevant
    {"query": "How to implement OAuth2 authentication?", "document": "OAuth2 is an authorization framework that enables applications to obtain limited access to user accounts", "expected": True},
    {"query": "What is Docker containerization?", "document": "Docker uses OS-level virtualization to deliver software in packages called containers", "expected": True},
    {"query": "Explain React hooks", "document": "React Hooks are functions that let you use state and other React features in functional components", "expected": True},
    {"query": "Python async/await tutorial", "document": "Asynchronous programming in Python with async and await keywords allows concurrent execution", "expected": True},
    {"query": "PostgreSQL indexing best practices", "document": "Database indexes improve query performance by providing quick access to rows in a table", "expected": True},
    
    # Technical queries - irrelevant
    {"query": "How to set up Kubernetes?", "document": "Chocolate chip cookies are a popular dessert made with flour, butter, and chocolate chips", "expected": False},
    {"query": "JavaScript promises explained", "document": "The Amazon rainforest is the world's largest tropical rainforest", "expected": False},
    {"query": "Git branching strategies", "document": "Tennis is a racket sport played between two players or two teams of two players", "expected": False},
    {"query": "Redis caching patterns", "document": "The Great Wall of China is a series of fortifications built across northern China", "expected": False},
    {"query": "GraphQL vs REST API", "document": "Photosynthesis is the process by which plants convert light energy into chemical energy", "expected": False},
    
    # Business queries - relevant
    {"query": "Customer retention strategies", "document": "Improving customer loyalty through personalized experiences and rewards programs increases retention", "expected": True},
    {"query": "Agile project management", "document": "Scrum is an agile framework for managing product development with iterative sprints", "expected": True},
    {"query": "Market segmentation techniques", "document": "Dividing a market into distinct groups of buyers with different needs and characteristics", "expected": True},
    {"query": "ROI calculation methods", "document": "Return on Investment measures the profitability of an investment relative to its cost", "expected": True},
    {"query": "Supply chain optimization", "document": "Streamlining logistics and inventory management to reduce costs and improve efficiency", "expected": True},
    
    # Business queries - irrelevant
    {"query": "Quarterly revenue forecasting", "document": "Penguins are flightless birds that live in the Southern Hemisphere", "expected": False},
    {"query": "Employee onboarding process", "document": "The speed of light in vacuum is approximately 299,792,458 meters per second", "expected": False},
    {"query": "Digital marketing strategies", "document": "Mount Everest is Earth's highest mountain above sea level", "expected": False},
    {"query": "Risk management framework", "document": "Pizza originated in Naples, Italy, and has become popular worldwide", "expected": False},
    {"query": "Competitive analysis methods", "document": "The human brain contains approximately 86 billion neurons", "expected": False},
    
    # Science queries - relevant
    {"query": "Climate change impacts", "document": "Global warming causes rising sea levels, extreme weather events, and ecosystem disruption", "expected": True},
    {"query": "CRISPR gene editing", "document": "CRISPR-Cas9 is a genome editing tool that allows researchers to alter DNA sequences", "expected": True},
    {"query": "Quantum computing principles", "document": "Quantum computers use quantum bits or qubits that can exist in superposition states", "expected": True},
    {"query": "Machine learning algorithms", "document": "Supervised learning uses labeled data to train models for classification and regression", "expected": True},
    {"query": "Renewable energy sources", "document": "Solar panels convert sunlight into electricity through photovoltaic cells", "expected": True},
    
    # Science queries - irrelevant
    {"query": "Black hole formation", "document": "Shakespeare wrote 37 plays and 154 sonnets during his lifetime", "expected": False},
    {"query": "DNA replication process", "document": "The FIFA World Cup is held every four years", "expected": False},
    {"query": "Artificial neural networks", "document": "Coffee beans are actually seeds from coffee plant berries", "expected": False},
    {"query": "Vaccine development stages", "document": "The Eiffel Tower was completed in 1889 for the Paris Exposition", "expected": False},
    {"query": "Biodiversity conservation", "document": "Bitcoin is a decentralized digital currency created in 2009", "expected": False},
    
    # Edge cases - ambiguous relevance
    {"query": "How to cook pasta?", "document": "Italian cuisine features many pasta dishes with various sauces and preparations", "expected": None},
    {"query": "Python programming", "document": "Snake handling requires careful safety precautions and proper equipment", "expected": None},
    {"query": "Cloud computing", "document": "Weather patterns are influenced by atmospheric pressure and temperature", "expected": None},
    {"query": "Java development", "document": "Coffee cultivation requires specific climate conditions and soil types", "expected": None},
    {"query": "Ruby on Rails", "document": "Precious gemstones are minerals valued for their beauty and rarity", "expected": None},
    
    # Very short queries/documents
    {"query": "API", "document": "Application Programming Interface", "expected": True},
    {"query": "ML", "document": "Machine Learning", "expected": True},
    {"query": "DB", "document": "Database", "expected": True},
    {"query": "UI", "document": "User Interface", "expected": True},
    {"query": "AI", "document": "Artificial Intelligence", "expected": True},
    
    # Long queries with short documents
    {"query": "What are the best practices for implementing microservices architecture in a distributed system?", "document": "Microservices guide", "expected": True},
    {"query": "How can I optimize database queries for better performance in a high-traffic application?", "document": "SQL optimization", "expected": True},
    {"query": "What security measures should be implemented when building a REST API?", "document": "API security", "expected": True},
    {"query": "How to implement continuous integration and deployment pipelines?", "document": "CI/CD tutorial", "expected": True},
    {"query": "What are the key considerations for building scalable web applications?", "document": "Scalability patterns", "expected": True},
    
    # Question variations
    {"query": "What is blockchain?", "document": "Distributed ledger technology for recording transactions", "expected": True},
    {"query": "How does blockchain work?", "document": "Recipes for homemade bread and pastries", "expected": False},
    {"query": "Why use blockchain?", "document": "Blockchain provides transparency and immutability for digital transactions", "expected": True},
    {"query": "When to use blockchain?", "document": "The history of ancient Egyptian pyramids", "expected": False},
    {"query": "Where is blockchain used?", "document": "Cryptocurrency, supply chain, and smart contracts utilize blockchain technology", "expected": True},
]

# Helper function to get test cases
def get_test_cases(count=None, relevance_filter=None):
    """
    Get test cases with optional filtering.
    
    Args:
        count: Number of test cases to return (None for all)
        relevance_filter: Filter by expected relevance (True, False, or None)
    
    Returns:
        List of test cases
    """
    cases = RANKING_TEST_CASES
    
    if relevance_filter is not None:
        cases = [case for case in cases if case["expected"] == relevance_filter]
    
    if count is not None:
        cases = cases[:count]
    
    return cases

# Example usage functions for different prompt formats
def format_for_ollama_generate(case, prompt_template=None):
    """Format a test case for Ollama generate API."""
    if prompt_template:
        return prompt_template.format(query=case["query"], document=case["document"])
    
    return f"""You are an expert relevance grader. Your task is to evaluate if the
    following document: {case["document"]} 
    
    is relevant to this question: {case["query"]}.

    You must ONLY answer yes or no."""

def format_for_binary_classification(case):
    """Format for simple binary classification."""
    return f"Query: {case['query']}\nDocument: {case['document']}\nRelevant? (yes/no):"

def format_for_score_output(case):
    """Format for numeric score output."""
    return f"Rate relevance (0-1):\nQuery: {case['query']}\nDocument: {case['document']}\nScore:"