"""Independent image-level metrics, separate from the DINO proxies.

The DINO projection, DINO CLS similarity and outside-ROI MSE used elsewhere are
all computed with the same encoder the adapter is trained against, so they
cannot testify that glasses were removed or that the person stayed the same.
These two scorers are independent of that encoder:

  GlassesScorer  - supervised ViT eyewear classifier.
  IdentityScorer - face-recognition embedding cosine.

Both are still imperfect: a classifier trained on other data can be wrong on
FLUX samples, and a recognition embedding measures "same identity" only as well
as its own training allows. Agreement with hand labels is measured and recorded
in out/detector_choice.json rather than assumed.
"""
import numpy as np
import torch

GLASSES_PROMPTS = [
    'a photo of a person wearing eyeglasses',
    'a portrait of someone with glasses on their face',
    'a face with spectacles',
]
NO_GLASSES_PROMPTS = [
    'a photo of a person without eyeglasses',
    'a portrait of someone with no glasses, bare eyes',
    'a face without spectacles',
]


GLASSES_MODEL = 'youngp5/eyeglasses_detection'
# Midpoint of the gap on 40 hand-labelled FLUX.1-schnell 512px portraits
# (out/detector_choice.json): min(glasses)=0.9843, max(no glasses)=0.0138.
GLASSES_THRESHOLD = 0.5


class GlassesScorer:
    """Supervised eyewear probability from a ViT glasses classifier.

    Chosen over CLIP zero-shot by measured agreement with hand labels: on the
    same 40 images full-frame CLIP ViT-L/14 separated the classes by 0.003 and
    was unusable, while this classifier separated them by 0.97 with no errors.
    """

    def __init__(self, model_name=GLASSES_MODEL, device='cuda:0', revision=None):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, revision=revision).to(device).eval()
        self.model.requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
        self.device = device
        self.model_name = model_name
        self.threshold = GLASSES_THRESHOLD
        labels = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        positive = [i for i, l in labels.items() if 'glass' in l and not l.startswith('no')]
        if len(positive) != 1:
            raise ValueError(f'Cannot identify the glasses class in {labels}')
        self.positive_index = positive[0]

    @torch.no_grad()
    def __call__(self, images):
        single = not isinstance(images, (list, tuple))
        batch = [im.convert('RGB') for im in ([images] if single else list(images))]
        inputs = self.processor(images=batch, return_tensors='pt').to(self.device)
        prob = self.model(**inputs).logits.softmax(-1)[:, self.positive_index]
        prob = prob.float().cpu().numpy()
        return float(prob[0]) if single else prob


class IdentityScorer:
    """Face-recognition embedding cosine similarity between two images.

    MTCNN crops and aligns the face, InceptionResnetV1 (VGGFace2) embeds it.
    deepface/ArcFace is unusable in this environment: its RetinaFace backend
    requires tf-keras, which conflicts with the installed TensorFlow 2.21.
    """

    def __init__(self, device='cuda:0'):
        from facenet_pytorch import InceptionResnetV1, MTCNN
        self.device = device
        self.detector = MTCNN(image_size=160, margin=20, post_process=True,
                              select_largest=True, device=device)
        self.model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        self.model.requires_grad_(False)
        self.model_name = 'facenet-pytorch InceptionResnetV1/vggface2 + MTCNN'
        self.error = None
        self.failures = 0

    @torch.no_grad()
    def embed(self, image):
        try:
            face = self.detector(image.convert('RGB'))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            face = None
        if face is None:
            self.failures += 1
            return None
        vec = self.model(face.unsqueeze(0).to(self.device))[0].double().cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    @staticmethod
    def similarity(a, b):
        if a is None or b is None:
            return None
        return float(np.dot(a, b))
