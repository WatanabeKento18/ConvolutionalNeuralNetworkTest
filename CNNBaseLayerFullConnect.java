package deeplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

public abstract class CNNBaseLayerFullConnect extends CNNBaseLayer{
	protected float[][][][] w;	//�d��(���̓m�[�h�ԍ�,���͉����W,���͏c���W,�o�̓m�[�h�ԍ�)
	protected float[] bias;	//�o�C�A�X
	
	protected float learnRate;	//�w�K��
	protected float[][][][] gradient;		//RMSProp�K�p�̂��߂̌��z�̓��̑��a
	protected float eps = 0.0001f;	//RMSProp�Ń[�����Z�����Ȃ����߂̏����Ȓl
	protected float rmsprop_alpha = 0.9f;	//RMSProp�̌��z�X�V�ɗp����l
	protected float lambda = 0.001f;		//�������̂��߂̏����Ȓ萔
	
	//�w���̕ۑ�(CNN�N���X��save���\�b�h���Ăяo��)
	public void save(BufferedWriter bw) throws IOException{
		int i , j , k , l;
		//�S�����w�̊�{������������
		bw.write(String.format("%d", layerCode));
		bw.newLine();
		bw.write(String.format("%d", sizeNin));
		bw.newLine();
		bw.write(String.format("%d", iSizeX));
		bw.newLine();
		bw.write(String.format("%d", iSizeY));
		bw.newLine();
		bw.write(String.format("%d", sizeNout));
		bw.newLine();
		bw.write(String.format("%f", learnRate));
		bw.newLine();
		//�t�B���^����������
		for( k = 0 ; k < sizeNin ; k++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					for( l = 0 ; l < sizeNout ; l++){
						bw.write(String.format("%f", w[k][i][j][l]));
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
		int i , j , k , l;
		//�d�݂�ǂݍ���
		for( k = 0 ; k < sizeNin ; k++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					for( l = 0 ; l < sizeNout ; l++){
						w[k][i][j][l] = Float.parseFloat(br.readLine());
					}
				}
			}	
		}
		//�o�C�A�X��ǂݍ���
		for( k = 0 ; k < sizeNout ; k++){
			bias[k] = Float.parseFloat(br.readLine());
		}
	}
	
	//���t�M����p�����t�`�d
	public abstract float[][][] backPropagation(float[][][] output , float[] label);
}