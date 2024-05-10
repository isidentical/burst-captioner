import time
import webdataset as wds
from argparse import ArgumentParser
from rich.progress import track
import braceexpand
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.messages import Response

PROMPT = """\
Can you please describe this image in up to two paragraphs? Please specify any objects within the image,
backgrounds, scenery, interactions, and gestures or poses. {alt_text} If they are multiple of
any object, please specify
how many. Is there text in the image, and if so, what does it say? If there is any lighting in the image, can you
identify where it is and what it looks like? What style is the image? If there are people or characters in the image,
what emotions are they conveying? Please keep your descriptions factual and terse but complete. DO NOT add any unnecessary
speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this
makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are
interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation.
Make sure to include the style of the image, for example cartoon, photograph, 3d render etc. Start with the words
‘This image showcases’:
"""

GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=300,
    repetition_penalty=1.2,
)

pipe = pipeline(
    "xtuner/llava-llama-3-8b-v1_1-hf",
    backend_config=TurbomindEngineConfig(session_len=8192, quant_policy=8),
    chat_template_config=ChatTemplateConfig(model_name="llama3"),
)


def recaption(images: list[Image.Image], metadata: list[dict[str, str]]) -> list[Response]:
    prompts = []
    for image, image_meta in zip(images, metadata):
        if caption := image_meta.get("caption"):
            alt_text = f"The original alt text includes {caption!r}."
        else:
            alt_text = ""

        prompts.append((PROMPT.format(alt_text=alt_text), image))

    response = pipe(prompts, gen_config=GENERATION_CONFIG)
    return response


def main():
    parser = ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("url")

    args = parser.parse_args()
    dataset = wds.DataPipeline(
        wds.SimpleShardList(braceexpand.braceexpand(args.url)),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("__key__", "jpg", "json"),
        wds.batched(batchsize=32),
    )

    t0 = t1 = time.perf_counter()
    for keys, jpgs, jsons in track(dataset):
        print("batch loading: ", time.perf_counter() - t1)
        t0 = time.perf_counter()
        captions = recaption(jpgs, jsons)
        t1 = time.perf_counter()
        total_toks = 0
        for caption in captions:
            print(caption.text)
            total_toks += caption.generate_token_len
        print("Took: ", t1 - t0, "to caption", len(keys))
        print("image/gpu/s: ", len(keys) / (t1 - t0))
        print("out tok/gpu/s:", total_toks / (t1 - t0))
        print("avg caption toks: ", total_toks / len(captions))
        continue


if __name__ == "__main__":
    main()
