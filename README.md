# DA6401 Assignment-2

This repo now contains a runnable baseline for the full Oxford-IIIT Pet pipeline:

- Task 1: `VGG11Classifier`
- Task 2: `VGG11Localizer`
- Task 3: `VGG11UNet`
- Task 4: `MultiTaskPerceptionModel`

It also includes Python experiment scripts for all eight Question 2 report items under `experiments/question2/`.

## Main Files

- `data/pets_dataset.py`: dataset loader for breed labels, bounding boxes, and trimaps
- `models/`: encoder, classifier, localizer, U-Net decoder, and multitask model
- `losses/iou_loss.py`: custom IoU loss
- `train.py`: training entrypoint for all tasks
- `inference.py`: single-image inference helper
- `experiments/question2/`: W&B-oriented scripts for Question 2

## Example Commands

```bash
python train.py --task classification --data-root oxford-iiit-pet --epochs 10 --checkpoint-path checkpoints/classifier.pth
python train.py --task localization --data-root oxford-iiit-pet --epochs 10 --encoder-checkpoint checkpoints/classifier.pth --checkpoint-path checkpoints/localizer.pth
python train.py --task segmentation --data-root oxford-iiit-pet --epochs 10 --encoder-checkpoint checkpoints/classifier.pth --checkpoint-path checkpoints/unet.pth
python train.py --task multitask --data-root oxford-iiit-pet --epochs 10 --classifier-checkpoint checkpoints/classifier.pth --localizer-checkpoint checkpoints/localizer.pth --segmentation-checkpoint checkpoints/unet.pth --checkpoint-path checkpoints/multitask.pth
```

```bash
python experiments/question2/q2_2_internal_dynamics.py --data-root oxford-iiit-pet --epochs 10
python experiments/question2/q2_3_transfer_learning_showdown.py --data-root oxford-iiit-pet --epochs 10
```

## Notes

- The dataset split is random but seed-controlled.
- Bounding boxes are normalized as `(x_center, y_center, width, height)`.
- Segmentation masks are converted to class ids `0, 1, 2`.
- `--encoder-checkpoint` lets Task 2 and Task 3 reuse Task 1 encoder weights.
- `--classifier-checkpoint`, `--localizer-checkpoint`, and `--segmentation-checkpoint` let Task 4 initialize from the individual task checkpoints.
- For Question 2.5, the script uses classification confidence as a practical confidence proxy when the multitask model is used.
