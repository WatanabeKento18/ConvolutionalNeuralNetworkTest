package deeplearning;

import java.util.Random;

/**
 * 
 * ��ݍ��݃j���[�����l�b�g���[�N�F��������ݍ��ݑw
 *
 */

public class CNNConvolutionActivation{
	int sizeN;					//�m�[�h��
	int iSizeX , iSizeY;		//���̓T�C�Y
	int sizeF;					//�t�B���^�T�C�Y
	int kernelNum;				//�J�[�l����
	public boolean topLayer;			//�擪���C���[�̏ꍇ��true
	
	private float[][][] filter;	//��ݍ��݃t�B���^
	private float[] bias;		//�o�C�A�X
	
	boolean activationMap[][][];		//�������ɂ��̗p���ꂽ���W��ۑ�
	
	private float[][][] beforeInput;	//���O�̓���
	
	float learnRate;
	
	CNNConvolutionActivation(int sizeN , int sizeX, int sizeY, int sizeF, int kernelNum){
		this.sizeN = sizeN;
		this.iSizeX = sizeX;
		this.iSizeY = sizeY;
		this.sizeF = sizeF;
		this.kernelNum = kernelNum;
		
		//����������
		Random rnd = new Random();
		
		//����������
		filter = new float[kernelNum][sizeF][sizeF];
		bias = new float[kernelNum];
		for( int kernel = 0 ; kernel < kernelNum ; kernel++){
			for( int i = 0 ; i < sizeF ; i++){
				for( int j = 0 ; j < sizeF ; j++){
					filter[kernel][i][j] = rnd.nextFloat();
				}
			}
		}for( int kernel = 0 ; kernel < kernelNum ; kernel++){
			bias[kernel] = 1.0f;
		}
		
		learnRate = 0.01f;
	}
	
	//���`�d
	//�����F���̓f�[�^�@�Ԃ�l�F�o�̓f�[�^
	float[][][] propagation(float[][][] input){
		int kernel , n , i , j , k , l;
		int oSizeX = getOutputSizeX();
		int oSizeY = getOutputSizeY();
		float max[] = new float[kernelNum];
		float min[] = new float[kernelNum];
		float[][][] output = new float[kernelNum][oSizeX][oSizeY];
		for( kernel = 0 ; kernel < kernelNum ; kernel++){
			max[kernel] = 0;
			min[kernel] = 0;
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					for( n = 0 ; n < sizeN ; n++){
						output[kernel][i][j] = 0;
						for( k = 0 ; k < sizeF ; k++){
							for( l = 0 ; l < sizeF ; l++){
								output[kernel][i][j] += input[n][i+k][j+l] * filter[kernel][k][l];
							}
						}
						output[kernel][i][j] += bias[kernel];
						//System.out.println("CONV SIZE:"+ sizeN + " i:"+i+" j:"+j + " DATA:"+output[kernel][i][j]);
					}
					if(max[kernel] < output[kernel][i][j]){
						max[kernel] = output[kernel][i][j];
					}else if(min[kernel] > output[kernel][i][j]){
						min[kernel] = output[kernel][i][j];
					}
				}
			}
		}
		for( kernel = 0 ; kernel < kernelNum ; kernel++){
			if(max[kernel] > 0.01f){
			//System.out.println("CONV SIZE:"+ sizeN + " MAX:"+max[kernel]);
				for( i = 0 ; i < oSizeX ; i++){
					for( j = 0 ; j < oSizeY ; j++){
						output[kernel][i][j] /= max[kernel];
						if(Float.isInfinite(output[kernel][i][j])){
							System.out.println("ERROR CONV SIZE:"+ sizeN + " DATA:"+output[kernel][i][j] + " MAX:" + max[kernel]);
							System.exit(0);
						}
					}
				}
			}else{
				float min_abs = min[kernel]*-1;
				for( i = 0 ; i < oSizeX ; i++){
					for( j = 0 ; j < oSizeY ; j++){
						output[kernel][i][j] = (output[kernel][i][j] + min_abs) / min_abs;
					}
				}
			}
		}
		//print(output);
		//�w�K�p�ɓ��͂��L��
		beforeInput = input;
		return output;
	}

	//�t�`�d
	//����:error[�m�[�h�ԍ�][����X��][����Y��] �O�̑w�̌덷�@input[�m�[�h�ԍ�][����X��][����Y��]�@���̑w�ɑ΂������
	public float[][][] backPropagation(float[][][] error){
		int kernel , n , i , j , k , l;
		int oSizeX = getOutputSizeX();
		int oSizeY = getOutputSizeY();
		float[][][] nextError = new float[sizeN][iSizeX][iSizeY];
		float sum;
		//�d�݂̊w�K
		for( n = 0 ; n < sizeN ; n++){
			for( k = 0 ; k < sizeF ; k++){
				for( l = 0 ; l < sizeF ; l++){
					for( kernel = 0 ; kernel < kernelNum ; kernel++){
						sum = 0;
						for( i = 0 ; i < oSizeX ; i++){
							for( j = 0 ; j < oSizeY ; j++){
								sum += error[kernel][i][j] * beforeInput[n][i+k][j+l];
							}
						}
						filter[kernel][k][l] -= sum*learnRate;
					}
				}
			}
		}
		//�o�C�A�X�̊w�K
		for( kernel = 0 ; kernel < kernelNum ; kernel++){
			sum = 0;
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					sum += error[kernel][i][j];
				}
			}
			bias[kernel] -= sum*learnRate;
		}
		//�d�݂̋t�`�d
		if(!topLayer){
			for( n = 0 ; n < sizeN ; n++){
				for( i = 0 ; i < iSizeX ; i++){
					for( j = 0 ; j < iSizeY ; j++){
						sum = 0;
						nextError[n][i][j] = 0;
						for( kernel = 0 ; kernel < kernelNum ; kernel++){
							for( k = 0 ; k < sizeF ; k++){
								for( l = 0 ; l < sizeF ; l++){
									if(i-k < 0 || j-l < 0 || i-k >= oSizeX || j-l >= oSizeY) continue;
									sum += error[kernel][i-k][j-l] * filter[kernel][k][l];
								}
							}
						}
						nextError[n][i][j] = sum;
						//System.out.println("NEXT ERR CONV:"+sum + " i:"+i+" j:"+j);
					}
				}
			}
		}
		return nextError;
	}
	
	//�o�͑wX�T�C�Y���擾
	public int getOutputSizeX(){
		return iSizeX - sizeF + 1;
	}
	
	//�o�͑wY�T�C�Y���擾
	public int getOutputSizeY(){
		return iSizeY - sizeF + 1;
	}
	
	//�o��
	public void print(float[][][] output){
		System.out.println("��ݍ��ݑw");
		for(int j = 0 ; j < getOutputSizeX() ; j++){
			for(int i = 0 ; i < getOutputSizeY() ; i++){
				System.out.print(String.format("%.2f",output[0][j][i]) + " ");
				/*if(output[0][i][j] > 0.2f) System.out.print("��");
				else System.out.print("��");*/
			}
			System.out.println();
		}
	}
	
}