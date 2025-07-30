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

async def main():
    await synthesizer.a_generate_goldens_from_scratch(num_goldens=5)

    df = synthesizer.to_pandas()
    df.to_csv("synthesizer_goldens_from_scratch.csv", index=False)

    # print(synthesizer.synthetic_goldens)


if __name__ == "__main__":
    asyncio.run(main())
