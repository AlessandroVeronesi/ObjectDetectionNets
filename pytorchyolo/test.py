#! /usr/bin/env python3

from __future__ import division

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config

# This function evaluates the performance of the model.
def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=16, img_size=640,
                        n_cpu=8, iou_thres=0.3, conf_thres=0.1, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names (e.g., ["cat", "dog", "car"])
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8 (Number of images processed at a time)
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416 (Resizes images to 416x416 pixels)
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """

    # This helper function is responsible for loading the image, resizing them to the img_size,
    # preparing batches of size batch_size and using n_cpu threads
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)

    # Load the model
    model = load_model(model_path, weights_path)

    # Evaluate the model
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output


# this function displays the metrics_output (evaluation results) in a readable format
def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        print('AP_CLASS', ap_class)
        print('AP:', len(AP))
        print('number of class names:', len(class_names))
        if verbose:
            # Create a table
            # ap_table = [
            #     ["Index", "Class", "AP"],
            #     [0, "Cat", "0.87654"],
            #     [1, "Dog", "0.82345"],
            #     [2, "Car", "0.91234"]
            # ]
            ap_table = [["Index", "Class", "AP"]]
            # Loops over the ap_class to keep a track of class_names and AP
            # ap_class is a list of indices representing the detected object classes
            # (e.g., [0, 1, 2] for "cat", "dog", and "car").
            # enumerate(ap_class) lets you loop over the indices (c) while also keeping track of their position (i).
            # for i, c in enumerate(ap_class):
            #     # Class index (c)
            #     # AP[i] AP value (formatted to 5 decimal places).
            #
            #     ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            valid_class_indices = [i for i in ap_class if i < 80]
            for c in valid_class_indices:
                    if c < len(class_names):  # Ensure valid index
                       ap_table += [[c, class_names[c], "%.5f" % AP[c]]]
                    else:
                       print(f"Skipping invalid class index {c}")
                # uses AsciiTable() to create a proper table
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.6f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

# This function evaluates a trained model's performance on validation datatset
# It calculates key metrics such as precision, recall, Average Precision (AP), F1 score, and mAP
def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of metrics
    #  true labels (targets)
    # tqdm.tqdm shows a progress bar to track the evaluation process.
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Take the class labels from the targets and store them in labels
        labels += targets[:, 1].tolist()
        # Convert bounding boxes from one format to another and adjust them to match the image size.
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        # Scale bounding boxes to image size
        targets[:, 2:] *= img_size

        # Convert them to tensors of the correct type and disable gradient calculations (since we are not training)
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            # Filter out redundant overlapping predictions to keep only the best ones
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        # get_batch_statistics() calculates:
        # True Positives (TP): Correctly predicted objects.
        # Confidence Scores: The model's confidence in its predictions.
        # Predicted Labels: The class labels predicted by the mode
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    # sample_metrics is a list of tuples and zip() unpacks all the values
    # and groups corresponding elements together.
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output


def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    print(f"Validation dataset paths: {img_path}")
    print(f"Number of images in validation dataset: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.3, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    # data config file contains paths to the training and validation data and
    # path to the file containing class names.
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names

    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)


if __name__ == "__main__":
    run()
