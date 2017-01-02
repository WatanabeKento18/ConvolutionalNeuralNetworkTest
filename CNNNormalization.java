package deeplearning;

import java.io.*;

/**
 * 
 * ��ݍ��݃j���[�����l�b�g���[�N�F��ݍ��ݑw
 *
 */

public class CNNNormalization extends CNNBaseLayer{

	int sizeF;	//�t�B���^�T�C�Y
	final int filterAreaSize;	//�Ǐ��t�B���^���m�[�h��
	final float divNormRange = 0.001f;
	
	//sizeNin:���̓m�[�h�� iSizeX:���̓f�[�^���T�C�Y iSizeY:���̓f�[�^�c�T�C�Y
	public CNNNormalization(int sizeNin , int iSizeX , int iSizeY, int sizeF){
			this.layerCode = 1051;
			this.sizeNin = sizeNin;
			this.sizeNout = sizeNin;
			this.iSizeX = iSizeX;
			this.iSizeY = iSizeY;
			this.oSizeX = iSizeX;
			this.oSizeY = iSizeY;
			this.sizeF = sizeF;
			if(sizeF%2 == 0){
				this.sizeF += 1;	//�t�B���^�T�C�Y����ɂ���
			}
			filterAreaSize = sizeNin*this.sizeF*this.sizeF;
	}
	
	public float[][][] propagation(float[][][] input) {
		int n, i, j, k, l, filterX, filterY;
		float subNormSum, divNormSum;
		float[][][] output = new float[sizeNout][oSizeX][oSizeY];
		final int filterStart = sizeF/(-2);
		final int filterEnd = sizeF/2;
		
		//���K��
		for(i = 0; i < iSizeX; i++){
			for(j = 0; j < iSizeY; j++){
				//���v�l���Z�o
				subNormSum = 0;
				divNormSum = 0;
				for(n = 0; n < sizeNin; n++){
					for(k = filterStart; k <= filterEnd; k++){
						for(l = filterStart; l <= filterEnd; l++){
							filterX = i+k;
							filterY = j+l;
							if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
								subNormSum += input[n][filterX][filterY]/filterAreaSize;
								divNormSum += input[n][filterX][filterY]*input[n][filterX][filterY]/filterAreaSize;
							}
						}
					}
				}
				//���Z���K���p
				divNormSum = (float)Math.sqrt((double)divNormSum);
				if(divNormSum < divNormRange){
					divNormSum = divNormRange;	//�l�̉�����ݒ�
				}
				//���K������
				for(n = 0; n < sizeNin; n++){
					output[n][i][j] = (input[n][i][j] - subNormSum)/divNormSum;
				}
			}
		}
		if(!useMinibatch) beforeInput = input;
		return output;
	}

	public float[][][] backPropagation(float[][][] error) {
		int ni, no, i, j, k, l, filterX, filterY;
		float subDif, divDif;
		float[][] divNormSum = new float[iSizeX][iSizeY];
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];
		final int filterStart = sizeF/(-2);
		final int filterEnd = sizeF/2;

		for(i = 0; i < iSizeX; i++){
			for(j = 0; j < iSizeY; j++){
				//���v�l���Z�o
				for(ni = 0; ni < sizeNin; ni++){
					for(k = filterStart; k <= filterEnd; k++){
						for(l = filterStart; l <= filterEnd; l++){
							filterX = i+k;
							filterY = j+l;
							if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
								divNormSum[i][j] += beforeInput[ni][filterX][filterY]*beforeInput[ni][filterX][filterY]/filterAreaSize;
							}
						}
					}
				}
				divNormSum[i][j] = (float)Math.sqrt((double)divNormSum[i][j]);
				if(divNormSum[i][j] < divNormRange){
					divNormSum[i][j] = divNormRange;	//�l�̉�����ݒ�
				}
			}
		}

		for(ni = 0; ni < sizeNin; ni++){
			for(i = 0; i < iSizeX; i++){
				for(j = 0; j < iSizeY; j++){
					for(no = 0; no < sizeNout; no++){
						for(k = filterStart; k <= filterEnd; k++){
							for(l = filterStart; l <= filterEnd; l++){
								filterX = i+k;
								filterY = j+l;
								if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
									divDif = 0;
									subDif = 0;
									if(k==0 && l==0 && ni==no){	//���͂Əo�͂��S���������W�����S�̏ꍇ
										divDif = 1.0f/divNormSum[filterX][filterY];
										subDif = 1.0f;
									}
									divDif -= beforeInput[ni][i][j]*beforeInput[ni][i][j]/
											(filterAreaSize*divNormSum[filterX][filterY]*divNormSum[filterX][filterY]*divNormSum[filterX][filterY]);
									subDif -= 1.0f/filterAreaSize; 
									nextError[ni][i][j] += subDif*divDif*error[no][filterX][filterY];
								}
							}
						}
					}
					//System.out.println(nextError[ni][i][j]);
				}
			}
		}
		if(Float.isNaN(nextError[0][0][0])) System.out.println("NORM1 EXCEPTION!!");
		if(Float.isNaN(error[0][0][0])) System.out.println("NORM2 EXCEPTION!!");
		//System.exit(0);
		return nextError;
	}

	public void training() {
	}

	public void save(BufferedWriter bw) throws IOException {
	}

	public void load(BufferedReader br) throws IOException {
	}
	
}