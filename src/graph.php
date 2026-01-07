<?php

use PHP2xAI\Runtime\PHP\Datasets\TrainValidateDataset;
use PHP2xAI\Runtime\PHP\Datasets\StreamFileDataset;
use PHP2xAI\Runtime\PHP\Optimizers\Adam;

ini_set('precision', 30);
ini_set('serialize_precision', 30);
ini_set("display_errors", "On");
ini_set("memory_limit", "10G");

include("../../autoload.php");
include("model.php");

$path = "./DataLabelInt/Training";
$pathVal = "./DataLabelInt/Test";
$outputPath = "./Output";

// Enable lightweight profiling of node creation and hot ops
// Profiler::enable();

if (!@is_dir($outputPath))
	@mkdir($outputPath, 0777, true);

$dataset = new StreamFileDataset($path."/train.txt", 300);
$valDataset = new StreamFileDataset($path."/test.txt", 300);

$tvDataset = new TrainValidateDataset($dataset, $valDataset);

$optimizer = new Adam(0.0001, 0.9, 0.999);
$optimizer->setGradClip(1.0); // evita spike di gradiente che fanno risalire la loss
$model = new MnistModel($optimizer, 128, 56);

$graph = $model->exportGrapf($tvDataset);

file_put_contents("./graph.json", json_encode($graph), LOCK_EX);