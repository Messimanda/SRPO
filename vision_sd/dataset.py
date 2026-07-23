from jinja2 import Template

from verl.utils.dataset import RLHFDataset


class VisionSDDataset(RLHFDataset):
    """RL dataset wrapper that keeps the raw question for teacher branches."""

    @staticmethod
    def _align_image_placeholders(prompt_str: str, n_images: int) -> str:
        """Ensure the number of <image> placeholders matches image count."""
        if n_images <= 0:
            return prompt_str.replace("<image>", "")

        n_tokens = prompt_str.count("<image>")
        if n_tokens == n_images:
            return prompt_str
        if n_tokens < n_images:
            return prompt_str + "<image>" * (n_images - n_tokens)

        # Too many placeholders: keep the first n_images and merge the remaining text.
        parts = prompt_str.split("<image>")
        kept = parts[0]
        for i in range(1, n_images + 1):
            kept += "<image>" + parts[i]
        kept += "".join(parts[n_images + 1 :])
        return kept

    def _build_messages(self, example: dict) -> list[dict]:
        """Build messages with image-token count aligned to actual images."""
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            images = example[self.image_key]
            if not isinstance(images, (list, tuple)):
                images = [images]
            prompt_str = self._align_image_placeholders(prompt_str, len(images))

            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})
                if content:
                    content_list.append({"type": "text", "text": content})
            return [{"role": "user", "content": content_list}]

        return super()._build_messages(example)

    def __getitem__(self, index):
        raw_example = self.dataset[index]
        raw_question = raw_example.get(self.prompt_key, raw_example.get("question", raw_example.get("problem")))
        example = super().__getitem__(index)
        example["question"] = raw_question
        example["uid"] = raw_example.get("uid", raw_example.get("id", raw_example.get("index", index)))
        return example
