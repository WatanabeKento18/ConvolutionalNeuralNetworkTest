package deeplearning;

import java.io.*;

/**
 * 		CNN�w�̊�{�ƂȂ�\��
**/

public abstract class CNNBaseLayer{
	protected int layerCode;			//�w�̎��ʔԍ�
	protected int sizeNin;				//���̓m�[�h��
	protected int iSizeX , iSizeY;		//���̓T�C�Y
	protected int sizeNout;				//�o�̓m�[�h��
	protected int oSizeX , oSizeY;		//�o�̓T�C�Y
	
	protected boolean useMinibatch;		//�~�j�o�b�`�w�K���s���ꍇtrue
	protected float[][][] beforeInput;				//�w�K�p���̓f�[�^
	
	public abstract float[][][] propagation(float[][][] input);
	public abstract float[][][] backPropagation(float[][][] error);
	public abstract void training();
	public abstract void save(BufferedWriter bw) throws IOException;
	public abstract void load(BufferedReader br) throws IOException;

	//�~�j�o�b�`�w�K�ݒ�
	public void initMinibatchSetting(int minibatchSize){
		useMinibatch = true;
	}
	
	//���̓T�C�Y�̔z����擾
	public float[][][] getInputClone(){
		return new float[sizeNin][iSizeX][iSizeY];
	}
	
	//�w�K�p�̓��̓f�[�^��ݒ�
	public void setInput(float[][][] input){
		beforeInput = input;
	}
	
	//�o�͑��̃T�C�Y���܂Ƃ߂Ď擾
	public int[] getOutputSizeList(){
		int[] sizeList = new int[3];
		sizeList[0] = sizeNout;
		sizeList[1] = oSizeX;
		sizeList[2] = oSizeY;
		return sizeList;
	}
}