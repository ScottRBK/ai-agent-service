from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.models import OllamaModel
import asyncio
styling_config = StylingConfig(
    input_format="Questions on current affairs, news, git hub repsitories",
    expected_output_format="""Responses should be accurate, up to date and relevant to the question.""",
    task="Answering information on current affairs, news and technology from using new sites, github, deepwiki and other relevant sources",
    scenario="Technical user speaking to their chat bot about current affairs, news and technology"
)

synthesizer = Synthesizer(
    model=OllamaModel(model="mistral:7b", temperature=0.0),
    styling_config=styling_config
)

tool_contexts = {
    "searxng_results": [
        "Search Results for 'AI regulation 2024':\n1. EU AI Act passes final vote...\n2. US considers federal AI framework...",
        "Latest developments show increased focus on safety and transparency requirements."
    ],
    "github_results": [
        "Repository: openai/whisper\nStars: 45k\nDescription: Robust speech recognition via large-scale weak supervision",
        "Recent activity: v3.0 release with improved multilingual support"
    ],
    "deepwiki_results": [
        "DeepWiki: Machine Learning Fundamentals\nSupervised learning uses labeled data to train models...",
        "Common algorithms include decision trees, neural networks, and support vector machines."
    ]
}

all_contexts = list(tool_contexts.values())

async def main():
    await synthesizer.a_generate_goldens_from_contexts(contexts=all_contexts)

    df = synthesizer.to_pandas()
    df.to_csv("synthesizer_goldens_from_context.csv", index=False)

    # print(synthesizer.synthetic_goldens)


if __name__ == "__main__":
    asyncio.run(main())
