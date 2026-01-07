<?php

use PHP2xAI\Models\Model;
use PHP2xAI\Tensor\Matrix;
use PHP2xAI\Tensor\Vector;
use PHP2xAI\Tensor\Scalar;
use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\PHP\Optimizers\Optimizer;

class MnistModel extends Model
{
	protected int $hidden1;
	protected int $hidden2;
	
	public function __construct(?Optimizer $optimizer = null, int $hidden1 = 128, int $hidden2 = 64)
	{
		$this->hidden1 = $hidden1;
		$this->hidden2 = $hidden2;
		
		$this->W1 = Matrix::init($hidden1, 784, 0.05);
		$this->W2 = Matrix::init($hidden2, $hidden1, 0.05);
		$this->W3 = Matrix::init(10, $hidden2, 0.05);
		
		$this->b1 = Vector::zeros($hidden1);
		$this->b2 = Vector::zeros($hidden2);
		$this->b3 = Vector::zeros(10);
		
		parent::__construct($optimizer);
	}
	
	public function forward(Tensor $x) : Tensor
	{
		$L1 = $this->W1->matmul($x)->add($this->b1)->ReLU();
		$L2 = $this->W2->matmul($L1)->add($this->b2)->ReLU();
		$L3 = $this->W3->matmul($L2)->add($this->b3);
		
		return $L3;
	}
	
	public function output(Tensor $x) : Tensor
	{
		$logits = $this->forward($x);
		
		return $logits->softmax();
	}
	
	public function loss(Tensor $x, Tensor $y) : Scalar
	{
		$logits = $this->forward($x);
		
		return $logits->CELogitsLabelInt($y);
	}
}
