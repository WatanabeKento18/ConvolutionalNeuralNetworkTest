package deeplearning;

import java.util.*;

/**
 * 
 * ��ݍ��݃j���[�����l�b�g���[�N�F�S�����w(�������֐�:SoftMax)
 *
 */

public class CNNFullConnectSoftMax extends CNNBaseLayerFullConnect{
	private float[][][] beforeError;	//���O�̌덷
	
	//sizeNin:���̓m�[�h�� sizeNout:�o�̓m�[�h�� iSizeX:���̓f�[�^���T�C�Y iSizeY:���̓f�[�^�c�T�C�Y
	public CNNFullConnectSoftMax(int sizeNin , int iSizeX , int iSizeY, int sizeNout ){
		this.layerCode = 1041;
		this.sizeNin = sizeNin;
		this.sizeNout = sizeNout;
		this.iSizeX = iSizeX;
		this.iSizeY = iSizeY;
		this.oSizeX = this.oSizeY = 1;
		
		//����������
		Random rnd = new Random();
		
		//������
		w = new float[sizeNin][iSizeX][iSizeY][sizeNout];
		for(int ni = 0 ; ni < sizeNin ; ni++){
			for(int i = 0 ; i < iSizeX ; i++){
				for(int j = 0 ; j < iSizeY ; j++){
					for(int no = 0 ; no < sizeNout ; no++){
						w[ni][i][j][no] = rnd.nextFloat() - 0.5f;
					}
				}				
			}			
		}
		bias = new float[sizeNout];
		for(int no = 0 ; no < sizeNout ; no++){
			bias[no] = 1.0f;
		}
		gradient = new float[sizeNin][iSizeX][iSizeY][sizeNout];
		
		learnRate = 0.001f;
	}
	
	//�w�K���ݒ�
	public void setLearningRate(float f){
		if(f < 0) f = 0.0f;
		else if(f > 1.0f) f = 1.0f;
		learnRate = f;
	}
	
	//���`�d�i�����@input[���̓m�[�h�ԍ�][X][Y]�F���̓f�[�^�@�Ԃ�l�@output[�o�̓m�[�h�ԍ�]�F�o�̓f�[�^�j
	public float[][][] propagation(float[][][] input){

		//�o�͕ϐ�����
		float[][][] output = new float[sizeNout][1][1];
		
		//�o�͂��v�Z
		int ni , no , i , j ;
		for(no = 0 ; no < sizeNout ; no++){
			output[no][0][0] = 0;
			for(ni = 0 ; ni < sizeNin ; ni++){
				//�d�݌v�Z
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						output[no][0][0] += input[ni][i][j] * w[ni][i][j][no];
					}
				}
			}
			output[no][0][0] += bias[no];	//�o�C�A�X���Z
			//System.out.println("PROP:"+output[no]);
		}
		output = softmax(output);
		
		//�w�K�p�ɓ��͂��L��
		if(!useMinibatch){
			beforeInput = input;
		}
		return output;
	}
	
	//�I���W�i��
	//�t�`�d(���� output[�m�[�h�ԍ�]:�o�͑w���̃f�[�^ input[]:���͑w���̃f�[�^ label[�m�[�h�ԍ�]�@�Ԃ�l error[�m�[�h�ԍ�][X][Y] ���̑w�̏d�݂�`�����덷)
	public float[][][] backPropagation(float[][][] output , float[] label){
		float[][][] error = new float[sizeNin][iSizeX][iSizeY];	//�덷�L���p
		float[] d = new float[sizeNout];
		
		int no, ni, i, j;
		float eperror;	//�w�K����Z��̌덷(�ꎞ�L���p)
		//�w�K
		for(no = 0 ; no < sizeNout ; no++){
			d[no] = output[no][0][0] - label[no];	//�o�͂ƃ��x���f�[�^�̌덷���Z�o
			//�d�ݗp���[�v
			//gradient[no] = rmsprop_alpha * gradient[no] + (1 - rmsprop_alpha)*d[no]*d[no];
			eperror = learnRate * d[no];
			for(ni = 0 ; ni < sizeNin ; ni++){
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						error[ni][i][j] += d[no] * w[ni][i][j][no];	//�`�d������덷���Z�o
						w[ni][i][j][no] -= eperror * beforeInput[ni][i][j];	//�d�݂��X�V
					}
				}
			}
			
			//�o�C�A�X���X�V
			bias[no] -= eperror;
		}
		return error;
	}

	//�I���W�i��
	//�t�`�d(���� beforeError[�o�̓m�[�h�ԍ�]:�O�w�̌덷 input[]:���͑w���̃f�[�^ label�@�Ԃ�l error[�m�[�h�ԍ�][X][Y] ���̑w�̏d�݂�`�����덷)
	public float[][][] backPropagation_(float[][][] beforeError){
		float[][][] error = new float[sizeNin][iSizeX][iSizeY];	//�덷�L���p
		
		int no , ni , i , j;
		float eperror;	//�w�K����Z��̌덷(�ꎞ�L���p)
		//�w�K
		for(no = 0 ; no < sizeNout ; no++){
			//�d�ݗp���[�v
			eperror = learnRate * beforeError[no][0][0];
			for(ni = 0 ; ni < sizeNin ; ni++){
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						error[ni][i][j] += beforeError[no][0][0] * w[ni][i][j][no];	//�`�d������덷���Z�o
						w[ni][i][j][no] -= eperror * beforeInput[ni][i][j];	//�d�݂��X�V
					}
				}
			}
			
			//�o�C�A�X���X�V
			bias[no] -= eperror;
		}
		return error;
	}

	// �I���W�i��(�~�j�o�b�`�p)
	// �t�`�d(���� beforeError[�o�̓m�[�h�ԍ�]:�O�w�̌덷 input[]:���͑w���̃f�[�^ label�@�Ԃ�l error[�m�[�h�ԍ�][X][Y] ���̑w�̏d�݂�`�����덷)
	public float[][][] backPropagation(float[][][] beforeError) {
		float[][][] error = new float[sizeNin][iSizeX][iSizeY]; // �덷�L���p

		int no, ni, i, j;
		// �w�K
		for (no = 0; no < sizeNout; no++) {
			// �d�ݗp���[�v
			for (ni = 0; ni < sizeNin; ni++) {
				for (i = 0; i < iSizeX; i++) {
					for (j = 0; j < iSizeY; j++) {
						error[ni][i][j] += beforeError[no][0][0]*w[ni][i][j][no]; // �`�d������덷���Z�o
						//if(Float.isNaN(error[ni][i][j])) System.out.println("FC ERR NAN:"+ni+":"+i+":"+j+":"+no);
					}
				}
			}
			
			//�o�C�A�X���X�V
			bias[no] -= learnRate * beforeError[no][0][0];
		}
		if(Float.isNaN(beforeError[0][0][0])) System.out.println("FC1 EXCEPTION!!");
		if(Float.isNaN(error[0][0][0])) System.out.println("FC2 EXCEPTION!!");
		this.beforeError = beforeError;
		return error;
	}
	
	//�p�����[�^����
	public void training(){
		int no , ni , i , j;
		float totalError;	//�ꎞ�L���p
		final float rmsprop_alpha2 = 1 - rmsprop_alpha;
		//�w�K
		for(no = 0 ; no < sizeNout ; no++){
			//�d�ݗp���[�v
			for(ni = 0 ; ni < sizeNin ; ni++){
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						totalError = beforeError[no][0][0] * beforeInput[ni][i][j];
						gradient[ni][i][j][no] = rmsprop_alpha*gradient[ni][i][j][no] + rmsprop_alpha2*totalError*totalError;
						w[ni][i][j][no] -= learnRate*totalError/(Math.sqrt(gradient[ni][i][j][no])+eps);	//�d�݂��X�V
						//if(Float.isNaN(totalError)) System.out.println("FC NAN:"+ni+":"+i+":"+j+":"+no);
					}
				}
			}
		}
	}
	
	//�d�݂��X�V(���� input:����)
	public void updateParameter(float[][][] input){
		
	}
	
	//softmax�֐�
	private float[][][] softmax(float[][][] input){
		int length = input.length;	//�z��
		float[][][] output = new float[length][1][1];
		float max = input[0][0][0];	//�ő�l�L���p
		float sum = 0;	//���v�l�L���p
		
		//���������̂��߂̑O����
		for(int i = 1 ; i < length ; i++){
			if(max < input[i][0][0]){
				max = input[i][0][0];
			}
			//System.out.println("MAX INPUT:"+max+" "+input[i]);
		}
		
		//���v�l���Z�o
		for(int i = 0 ; i < length ; i++){
			input[i][0][0] -= max;
			sum += Math.exp(input[i][0][0]);
		}

		for(int i = 0 ; i < length ; i++){
			output[i][0][0] = (float)Math.exp(input[i][0][0])/sum;
		}
		if(Float.isNaN(output[0][0][0])){
			System.out.println("�Ȃ����G���[���o�܂��� : "+sum);
			for(int no = 0 ; no < sizeNout ; no++){
				System.out.println("["+no+"]:"+input[no][0][0]);
			}
			//System.exit(0);
		}
		return output;
	}
	
	//�e�X�g�p
	public static void main(String[] args){
		float[][][] train ={
				{	{1.0f , 1.0f , 0.0f},
					{1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 1.0f , 0.0f},
					{1.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 1.0f},
					{0.0f , 1.0f , 1.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 1.0f},
					{0.0f , 1.0f , 0.0f},}
		};
		
		float[][] label = {
				{1.0f,0.0f},
				{1.0f,0.0f},
				{0.0f,1.0f},
				{0.0f,1.0f},
		};
		
		float[][][] test ={
				{	{1.0f , 1.0f , 0.0f},
					{1.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 1.0f , 0.0f},
					{1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 1.0f},
					{0.0f , 1.0f , 1.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 1.0f},
					{0.0f , 1.0f , 0.0f},}
		};
		
		CNNFullConnectSoftMax fcl = new CNNFullConnectSoftMax(1,3,3,2);
		for(int epoch = 0 ; epoch < 1000 ; epoch++){
			for(int data = 0 ; data < 4 ; data++){
				float[][][] input = new float[1][3][3];
				for(int i = 0 ; i < 3 ; i++){
					for(int j = 0 ; j < 3 ; j++){
						input[0][i][j] = train[data][i][j];
					}
				}
				float[][][] output = fcl.propagation(input);
				fcl.backPropagation(output, label[data]);
			}
			System.out.println("TRAINING EPOCH :" + epoch + " / 1000");
		}
		
		System.out.println("---TEST---");
		for(int data = 0 ; data < 4 ; data++){
			System.out.println("DATA:"+data);
			float[][][] input = new float[1][3][3];
			for(int i = 0 ; i < 3 ; i++){
				for(int j = 0 ; j < 3 ; j++){
					input[0][i][j] = test[data][i][j];
				}
			}
			float[][][] output = fcl.propagation(input);
			System.out.println("CLASS1:"+output[0][0][0]+" CLASS2:"+output[1][0][0]);
		}
	}
}