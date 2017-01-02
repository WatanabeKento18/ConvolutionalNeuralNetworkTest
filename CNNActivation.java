package deeplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

/**
 * 
 * ��ݍ��݃j���[�����l�b�g���[�N�F�������w
 *
 */
public class CNNActivation extends CNNBaseLayer{
	boolean activationMap[][][];		//���������ꂽ���W��ۑ�
	
	CNNActivation(int sizeN , int sizeX , int sizeY){
		this.layerCode = 1021;
		this.sizeNin = this.sizeNout = sizeN;
		this.iSizeX = this.oSizeX = sizeX;
		this.iSizeY = this.oSizeY = sizeY;
		activationMap = new boolean[sizeN][sizeX][sizeY];		
	}
	
	
	//���`�d(�������֐�:ReLU)
	public float[][][] propagation(float[][][] input){
		int n , i , j;
		float[][][] output = new float[sizeNin][iSizeX][iSizeY];
		for( n = 0 ; n < sizeNin ; n++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					if(input[n][i][j] > 0){
						output[n][i][j] = input[n][i][j];
						activationMap[n][i][j] = true;
					}else{
						output[n][i][j] = 0;
						activationMap[n][i][j] = false;
					}
				}
			}
		}
		return output;
	}
	
	//�t�`�d
	//����:[�m�[�h�ԍ�][����X��][����Y��]
	public float[][][] backPropagation(float[][][] error){
		int n , i , j;
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];
		for( n = 0 ; n < sizeNin ; n++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					if(activationMap[n][i][j]){
						nextError[n][i][j] = error[n][i][j];
					}else{
						nextError[n][i][j] = 0;
					}
					//System.out.println("NEXT ERR ACTV:"+nextError[n][i][j] + " i:"+i+" j:"+j);
				}
			}
		}
		return nextError;
	}
	
	public void training(){		
	}
	
	//�w���̕ۑ�(CNN�N���X��save���\�b�h���Ăяo��)
	public void save(BufferedWriter bw) throws IOException{
		//�������w�̊�{������������
		bw.write(String.format("%d", layerCode));
		bw.newLine();
		bw.write(String.format("%d", sizeNin));
		bw.newLine();
		bw.write(String.format("%d", iSizeX));
		bw.newLine();
		bw.write(String.format("%d", iSizeY));
		bw.newLine();
	}
	
	//�w���̓ǂݍ���(CNN�N���X��load���\�b�h���Ăяo��)
	public void load(BufferedReader br) throws IOException{
	
	}
		
	//�e�X�g�p
	public static void main(String[] args){
		float[][][] image = {{{-0.5f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.1f , 0.9f},
							{0.9f , 0.5f , 0.0f , 0.1f , 0.0f , 0.0f , 0.9f , 0.9f , 0.9f},
							{-0.9f , 0.9f , 0.1f , 0.0f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{0.1f , 0.9f , 0.1f , -0.1f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{-0.1f , 0.9f , 0.9f , 0.1f , 0.0f , 0.9f , 0.9f , 0.9f , 0.1f},
							{0.0f , -0.1f , 0.9f , -0.9f , 0.9f , 0.1f , 0.2f , 0.9f , 0.1f}}};
		System.out.println("�����������O");
		for(int j = 0 ; j < 6 ; j++){
			for(int i = 0 ; i < 9 ; i++){
				System.out.print(image[0][j][i] + " ");
			}
			System.out.println();
		}
		System.out.println("������������");
		CNNActivation actv = new CNNActivation(1,6,9);
		float[][][] output = actv.propagation(image);
		for(int i = 0 ; i < output[0].length ; i++){
			for(int j = 0 ; j < output[0][0].length ; j++){
				System.out.print(String.format("%.1f",output[0][i][j]) + " ");
			}
			System.out.println();
		}
	}
}