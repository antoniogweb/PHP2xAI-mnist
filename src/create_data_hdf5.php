<?php
ini_set("display_errors", "On");
ini_set("memory_limit", "10G");

use PHP2xAI\Runtime\PHP\Datasets\PHP2XAIHDF5;
use PHP2xAI\Utility\Images\Vectorizer;

include("../vendor/autoload.php");

mt_srand(42);

$trainingSplit = 0.8;
$numberOfElements = 28 * 28; // 784

$imagesPath = __DIR__ . "/images";
$trainingPath = __DIR__ . "/DataLabelInt/Training";
$testPath = __DIR__ . "/DataLabelInt/Test";

ensureDataPath($trainingPath);
ensureDataPath($testPath);

$trainingFiles = [];
$testFiles = [];

for ($digit = 0; $digit <= 9; $digit++)
{
	$digitPath = $imagesPath . "/" . $digit;
	if (!is_dir($digitPath))
		continue;

	$digitFiles = glob($digitPath . "/*.png") ?: [];
	if (count($digitFiles) === 0)
		continue;

	shuffle($digitFiles);
	$trainingCount = (int)round(count($digitFiles) * $trainingSplit);

	foreach (array_slice($digitFiles, 0, $trainingCount) as $file)
		$trainingFiles[] = ["digit" => $digit, "path" => $file];

	foreach (array_slice($digitFiles, $trainingCount) as $file)
		$testFiles[] = ["digit" => $digit, "path" => $file];
}

shuffle($trainingFiles);
shuffle($testFiles);

writeHDF5($trainingFiles, $trainingPath . "/train.h5", $numberOfElements);
writeHDF5($testFiles, $trainingPath . "/test.h5", $numberOfElements);

/** Ensure an output directory exists. */
function ensureDataPath(string $path): void
{
	if (!is_dir($path) && !mkdir($path, 0777, true) && !is_dir($path))
		throw new RuntimeException("Unable to create data directory: {$path}");
}

/** Write image vectors and integer labels to an HDF5 file. */
function writeHDF5(array $files, string $targetPath, int $numberOfElements): void
{
	$dataset = PHP2XAIHDF5::create($targetPath);
	try {
		$dataset->setField('x', PHP2XAIHDF5::FLOAT32, [$numberOfElements]);
		$dataset->setField('y', PHP2XAIHDF5::INT64, [1]);

		foreach ($files as $fileInfo)
		{
			$vector = new Vectorizer($fileInfo["path"]);
			$x = $vector->toArray(true);
			if (count($x) !== $numberOfElements)
				throw new RuntimeException("Unexpected image vector size in {$fileInfo['path']}");

			$dataset->add('x', $x);
			$dataset->add('y', [(int)$fileInfo["digit"]]);
		}
	}
	finally {
		$dataset->destroy();
	}
}
