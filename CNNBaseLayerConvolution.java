package deeplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

public abstract class CNNBaseLayerConvolution extends CNNBaseLayer{
	protected int sizeF;			//�t�B���^�T�C�Y
	
	protected float[][][][] filter;	//��ݍ��݃t�B���^
	protected float[] bias;			//�o�C�A�X
	protected float learnRate; 		//�w�K��
	
	protected float[][][][] gradient;		//RMSProp�K�p�̂��߂̌��z�̓��̑��a
	protected float[] gradient_bias;		//�o�C�A�X�p
	protected float eps = 0.0001f;	//RMSProp�Ń[�����Z�����Ȃ����߂̏����Ȓl
	protected float rmsprop_alpha = 0.9f;	//RMSProp�̌��z�X�V�ɗp����l
	protected float lambda = 0.001f;		//�������̂��߂̏����Ȓ萔
	
	//�w���̕ۑ�(CNN�N���X��save���\�b�h���Ăяo��)
	public void save(BufferedWriter bw) throws IOException{
		int n, k, i, j;
		//��ݍ��ݑw�̊�{������������
		bw.write(String.format("%d", layerCode));
		bw.newLine();
		bw.write(String.format("%d", sizeNin));
		bw.newLine();
		bw.write(String.format("%d", iSizeX));
		bw.newLine();
		bw.write(String.format("%d", iSizeY));
		bw.newLine();
		bw.write(String.format("%d", sizeF));
		bw.newLine();
		bw.write(String.format("%d", sizeNout));
		bw.newLine();
		bw.write(String.format("%f", learnRate));
		bw.newLine();
		//�t�B���^����������
		for( n = 0; n < sizeNin; n++){
			for( k = 0 ; k < sizeNout ; k++){
				for( i = 0 ; i < sizeF ; i++){
					for( j = 0 ; j < sizeF ; j++){
						bw.write(String.format("%f", filter[n][k][i][j]));
						bw.newLine();
					}
				}
			}
		}
		//�o�C�A�X����������
		for( k = 0 ; k < sizeNout ; k++){
			bw.write(String.format("%f", bias[k]));
			bw.newLine();
		}
	}
	
	//�w���̓ǂݍ���(CNN�N���X��load���\�b�h���Ăяo��)
	public void load(BufferedReader br) throws IOException{
		int n, k, i, j;
		//�t�B���^��ǂݍ���
		for( n = 0; n < sizeNin; n++){
			for( k = 0 ; k < sizeNout ; k++){
				for( i = 0 ; i < sizeF ; i++){
					for( j = 0 ; j < sizeF ; j++){
						filter[n][k][i][j] = Float.parseFloat(br.readLine());
					}
				}
			}
		}
		//�o�C�A�X��ǂݍ���
		for( k = 0 ; k < sizeNout ; k++){
			bias[k] = Float.parseFloat(br.readLine());
		}
	}
}