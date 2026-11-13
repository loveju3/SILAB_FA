"""Synthetic neural loss/checkpoint tests; no history or dev records."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))


@unittest.skipUnless(importlib.util.find_spec('torch'), '需 embedding 依賴')
class NeuralTrainingTest(unittest.TestCase):
    def test_known_and_unknown_losses_update_encoder_and_heads(self):
        import torch
        from triage.neural import NeuralHeads
        from training.finetuning import loss_for_batch
        torch.manual_seed(42)
        encoder = torch.nn.Linear(3, 8)
        heads = NeuralHeads(8, .2)
        before = encoder.weight.detach().clone()
        optimizer = torch.optim.AdamW([*encoder.parameters(), *heads.parameters()], lr=.001)
        features = torch.randn(4, 3)
        labels = torch.tensor([0, -1, 3, -1])
        cat, info = heads(encoder(features))
        loss = loss_for_batch(cat, info, labels, (labels >= 0).long(), torch.ones(5), torch.ones(2), .1)
        loss.backward()
        self.assertTrue(torch.isfinite(encoder.weight.grad).all())
        self.assertGreater(heads.category.weight.grad.abs().sum().item(), 0)
        self.assertGreater(heads.information.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, encoder.weight))
        # Null-only batches must be finite, backward-capable, and train information.
        optimizer.zero_grad(set_to_none=True)
        cat, info = heads(encoder(features))
        loss = loss_for_batch(cat, info, torch.full((4,), -1), torch.zeros(4, dtype=torch.long),
                              torch.ones(5), torch.ones(2), .1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(heads.information.weight.grad.abs().sum().item(), 0)

    def test_early_stopping_keeps_first_best_checkpoint(self):
        import numpy as np
        import torch
        from training.finetuning import train_neural
        from training.settings import FineTuneSettings
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Module()
                self.embeddings.word_embeddings = torch.nn.Embedding(24, 8)
        class Encoder:
            def __init__(self, path):
                self.model = Model().eval()
                self.dimension = 8
        labels = ['noise', 'temperature', 'power', 'software', 'other', None] * 3
        vectors = np.ones((18, 8), dtype=np.float32)
        score = {'six_class_macro_f1': .5, 'category_macro_f1': .6}
        settings = FineTuneSettings(epochs=8, patience=2)
        with tempfile.TemporaryDirectory() as temporary, \
             patch('training.finetuning.BgeEncoder', Encoder), \
             patch('training.finetuning.evaluate_heads', return_value=(score, score)), \
             patch('training.finetuning.save_checkpoint') as save:
            result = train_neural(Path(temporary), [str(i) for i in range(18)], labels,
                                  list(range(12)), list(range(12, 18)), vectors, settings, {},
                                  Path(temporary), 'frozen-neural', {}, lambda *a, **kw: None)
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['epochs_completed'], 3)
            self.assertFalse(result['embedding_weights_changed'])
            self.assertEqual(save.call_count, 1)
            self.assertEqual(save.call_args.args[-1]['best_epoch'], 1)

    def test_neural_checkpoint_roundtrip_and_hash_validation(self):
        import json
        import numpy as np
        import torch
        from safetensors.torch import save_file
        from triage.neural import NeuralHeads, NeuralBgeClassifier, CLASSES
        from triage.io import file_sha256
        from training.settings import FineTuneSettings
        from dataclasses import asdict
        class Encoder:
            dimension = 8
            def transform(self, texts):
                return np.ones((len(texts), 8), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            heads = NeuralHeads(8, .2).eval()
            save_file(heads.state_dict(), str(folder / 'heads.safetensors'))
            metadata = {'format': 'bge-neural-v1', 'classes': list(CLASSES),
                        'settings': asdict(FineTuneSettings()), 'heads_sha256': file_sha256(folder / 'heads.safetensors')}
            (folder / 'model.json').write_text(json.dumps(metadata))
            with patch('triage.neural.BgeEncoder', return_value=Encoder()):
                loaded, _ = NeuralBgeClassifier.load(folder, {'E-01': {'category': 'power'}})
                batch = loaded.classify_many(['測試文字', 'E-01'])
                self.assertEqual(batch.predictions[1].category, 'power')
                self.assertEqual(batch.predictions[0], loaded.classify('測試文字'))
                with torch.inference_mode():
                    cat, info = heads(torch.ones(1, 8))
                expected = None if info.argmax(-1).item() == 0 else CLASSES[cat.argmax(-1).item()]
                self.assertEqual(batch.predictions[0].category, expected)
                (folder / 'heads.safetensors').write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError, '雜湊'):
                    NeuralBgeClassifier.load(folder, {})


if __name__ == '__main__':
    unittest.main()
