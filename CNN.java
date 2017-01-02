package deeplearning;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * 
 *	��ݍ��݃j���[�����l�b�g���[�N
 *	CNNConvolution,CNNActivation,CNNPooling,CNNFullConnect���\��
 *
**/

public class CNN{
	private ArrayList<CNNBaseLayer> layerList;	//CNN���\������w�̃��X�g
	private int convLayerNum;				//��ݍ��ݑw�̐�
	private int count;

	//�w�K�󋵊m�F��ʂɊւ���ݒ�
	private boolean useLearningDisplay;		//�w�K�󋵊m�F��ʂ𗘗p���邩�ǂ���
	private LearningDisplay learningDisplay;	//�w�K�󋵊m�F���
	private int correctNum;					//�w�K���x�����p
	
	//�w�K���f���ۑ��p�����[�^
	private boolean isModelSave;
	private String modelFilePath;
	
	//�w�K�p�ɕێ����Ă����p�����[�^
	private int[] inputSize;	//���͑w�T�C�Y
	private int trainingEpochs;	//�w�K��
	private int trainDataNum;	//�w�K�f�[�^��
	private float learnRate;		//�w�K��
	
	enum LayerCode{
		CONVOLUTION , ACTIVATION , POOLING , FULLCONNECT
	};
	
	public CNN(){
		layerList = new ArrayList<CNNBaseLayer>();
		convLayerNum = 0; 
	}
	
	//CNN�̏�Ԃ����Z�b�g����
	public void reset(){
		layerList.clear();
		convLayerNum = 0;
	}
	
	//�w�K���f���ۑ��Ɋւ���ݒ�
	public void setModelSave(String modelFilePath){
		if(modelFilePath == null){
			isModelSave = false;
			return;
		}
		this.modelFilePath = modelFilePath;
		isModelSave = true;
	}
	
	//�w�K���f����ۑ�����
	public void save(String fileName) {
		int layerSize = layerList.size();
		CNNBaseLayer layer;
		try {
			// �t�@�C���o�̓X�g���[��
			BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
			bw.write(String.format("%d", layerList.size()));	//�w�̐�
			bw.newLine();
			
			//�e�w�̏�����������
			for(int i = 0 ; i < layerSize ; i++){
				layer = layerList.get(i);
				layer.save(bw);
			}
			// �t�@�C���N���[�Y
			bw.close();
			System.out.println("�w�K���f�����t�@�C��[" + fileName + "]�ŕۑ����܂���");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//�w�K���f������CNN���\�z����
	public void load(String fileName){
		reset();
		int sizeNin , sizeNout , sizeX , sizeY , sizeF , colorNum;
		CNNBaseLayer layer;
		try {
			//�t�@�C�����̓X�g���[��
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			int layerSize = Integer.parseInt(br.readLine());	//�w�̐�
			
			//�e�w�����Ԃɓǂݍ���
			for(int i = 0 ; i < layerSize ; i++){
				//�w�̎��ʔԍ����m�F
				int layerType = Integer.parseInt(br.readLine());	//�w�̐�
				if(layerType < 1000){
					System.out.println("�w�K���f���ǂݍ��݃G���[");
					return;
				}
				
				//�w���Ƃɓǂݍ���
				switch(layerType){
				//��ݍ��ݑw
				case 1011:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					sizeF = Integer.parseInt(br.readLine());
					sizeNout = Integer.parseInt(br.readLine());
					learnRate = Float.parseFloat(br.readLine());
					addConvolutionLayer(sizeNin , sizeX , sizeY , sizeF , sizeNout);	//�V�����w��o�^
					layer = layerList.get(layerList.size()-1);
					layer.load(br);
					break;
				//�������w
				case 1021:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					addActivationLayer(sizeNin , sizeX , sizeY);	//�V�����w��o�^
					break;
				//�v�[�����O�w
				case 1031:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					addPoolingLayer(sizeNin , sizeX , sizeY);	//�V�����w��o�^
					break;
				//�S�����w
				case 1041:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					sizeNout = Integer.parseInt(br.readLine());
					learnRate = Float.parseFloat(br.readLine());
					addFullConnectLayer(sizeNin , sizeX , sizeY , sizeNout);	//�V�����w��o�^
					layer = layerList.get(layerList.size()-1);
					layer.load(br);
					break;
				}
			}
			//�t�@�C���N���[�Y
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//�w�K�󋵊m�F��ʂ��N��
	public void startLearningDisplay(){
		learningDisplay = new LearningDisplay();
		learningDisplay.setup();
		learningDisplay.setParameter(trainingEpochs, trainDataNum);
		learningDisplay.startTraining();
	}
	
	//���̓f�[�^�T�C�Y���`
	public void setInputSize(int color, int x, int y){
		inputSize = new int[3];
		inputSize[0] = color;
		inputSize[1] = x;
		inputSize[2] = y;
	}
	
	//��ݍ��ݑw��ǉ�
	//(nodeNum:���̓m�[�h�� inputSizeX:���̓f�[�^X�T�C�Y inputSizeY:���̓f�[�^Y�T�C�Y filterSize:�t�B���^�� kernelNum:�t�B���^�̐�)
	public void addConvolutionLayer(int nodeNum , int inputSizeX , int inputSizeY , int filterSize , int kernelNum ){
		boolean isTop = false;
		if(convLayerNum == 0)	isTop = true;	//��ݍ��ݑw�̒��Ő擪�̏ꍇ
		layerList.add(new CNNConvolution(nodeNum,inputSizeX,inputSizeY,filterSize,kernelNum,isTop));
		convLayerNum++;
	}
	//�p�����[�^�������Ŕ��ʂ��ď�ݍ��ݑw��ǉ�
	//(filterSize:�t�B���^�� kernelNum:�t�B���^�̐�)
	public void addConvolutionLayer(int filterSize , int kernelNum){
		boolean isTop = false;
		if(convLayerNum == 0)	isTop = true;	//��ݍ��ݑw�̒��Ő擪�̏ꍇ
		int[] parameter = getBeforeLayerParameter();
		layerList.add(new CNNConvolution(parameter[0],parameter[1],parameter[2],filterSize,kernelNum,isTop));
		convLayerNum++;
	}
	
	//�������w��ǉ�
	//(nodeNum:���̓m�[�h�� inputSizeX:���̓f�[�^X�T�C�Y inputSizeY:���̓f�[�^Y�T�C�Y filterSize:�t�B���^�� kernelNum:�t�B���^�̐�)
	public void addActivationLayer(int nodeNum , int inputSizeX , int inputSizeY){
		layerList.add(new CNNActivation(nodeNum,inputSizeX,inputSizeY));
	}
	//�p�����[�^�������Ŕ��ʂ��Ċ������w��ǉ�
	public void addActivationLayer(){
		addLayer(1021);
	}
	
	//�v�[�����O�w��ǉ�
	//(nodeNum:���̓m�[�h�� inputSizeX:���̓f�[�^X�T�C�Y inputSizeY:���̓f�[�^Y�T�C�Y filterSize:�t�B���^�� kernelNum:�t�B���^�̐�)
	public void addPoolingLayer(int nodeNum , int inputSizeX , int inputSizeY){
		layerList.add(new CNNPooling(nodeNum,inputSizeX,inputSizeY));
	}
	//�p�����[�^�������Ŕ��ʂ��ăv�[�����O�w��ǉ�
	public void addPoolingLayer(){
		addLayer(1031);
	}
	
	//�S�����w1��ǉ�
	//(nodeNum:���̓m�[�h�� inputSizeX:���̓f�[�^X�T�C�Y inputSizeY:���̓f�[�^Y�T�C�Y outputSize:�o�̓T�C�Y)
	public void addFullConnectLayer(int nodeNum , int inputSizeX , int inputSizeY , int outputSize){
		layerList.add(new CNNFullConnectSoftMax(nodeNum,inputSizeX,inputSizeY,outputSize));
	}
	//�p�����[�^�������Ŕ��ʂ��đS�����w��ǉ�
	//(outputSize:�o�̓T�C�Y)
	public void addFullConnectLayer(int outputSize){
		addLayer(1041, outputSize);
	}
	
	//�R�[�h���w�肵�đw��ǉ�
	public void addLayer(int code, int parameter1, int parameter2){
		int[] parameter = getBeforeLayerParameter();
		switch(code){
		//��ݍ��ݑw �p�����[�^1:�t�B���^�� �p�����[�^2:�t�B���^�T�C�Y
		case 1011:
			layerList.add(new CNNConvolution(parameter[0],parameter[1],parameter[2],parameter2,parameter1,true));
			break;
		//�������w
		case 1021:
			layerList.add(new CNNActivation(parameter[0],parameter[1],parameter[2]));
			break;
		//�v�[�����O�w
		case 1031:
			layerList.add(new CNNPooling(parameter[0],parameter[1],parameter[2]));
			break;
		//�S�����w �p�����[�^1:�o�͎�����
		case 1041:	//�S����+SoftMax�w
			layerList.add(new CNNFullConnectSoftMax(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		case 1042:	//�S����+ReLU�w
			layerList.add(new CNNFullConnectReLU(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		case 1043:	//�S��������������
			layerList.add(new CNNFullConnectNone(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		//���K���w �p�����[�^1:�t�B���^�T�C�Y
		case 1051:	//�Ǐ��R���g���X�g���K��
			layerList.add(new CNNNormalization(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		case 1052:	//���Z���K��
			layerList.add(new CNNSubNormalization(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		}
	}
	public void addLayer(int code){
		addLayer(code, 0, 0);
	}
	public void addLayer(int code, int parameter){
		addLayer(code, parameter, 0);
	}
	
	//���`�d
	public float[][][] propagation(float[][][] input){
		int layerSize = layerList.size();	//�w�̐�
		CNNBaseLayer layer;
		float[][][] output = input;
		/*float[] output2 = null;
		boolean convolutionLayerEnd = false;	//�S�����w�܂œ`�d������true*/
		
		//�ŏ��̑w����S�����w�܂ł�`�d
		for(int i = 0 ; i < layerSize ; i++){
			layer = layerList.get(i);
			output = layer.propagation(output);
		}
		
		//�S�����w����Ō�̑w�܂œ`�d
		/*for(; layer < layerSize ; layer++){
			layerCode = layerList.get(layer);
			layerType = layerCode/100;
			layerNumber = layerCode%100;
			
			if(layerType != 5) return null;	//�S�����w2�ł͂Ȃ��ꍇ�͕s��
			CNNFullConnect2 fc = fc2Layer.get(layerNumber);
			output2 = fc.propagation(output2);
		}
		return output2;*/
		return output;
	}

	//�t�`�d
	private void backPropagation(float[][][] output , float[] label){
		int layerSize = layerList.size();	//�w�̐�
		//�e�w�̏���ێ�
		int i = layerSize-1;	//�Q�Ƃ���w
		CNNBaseLayer layer;
		float[][][] error = null;		//�`�d������덷
		
		//�ŏI�w���S�����w�ɂȂ��Ă��邩���m�F�����t�M������덷���Z�o
		layer = layerList.get(i);
		if(layer instanceof CNNBaseLayerFullConnect){
			error = ((CNNBaseLayerFullConnect)layer).backPropagation(output , label);
		}
		
		//�S�����w����ŏ��̑w�܂Ō덷��`�d
		for( i = layerSize - 2 ; i >= 0 ; i--){
			layer = layerList.get(i);
			error = layer.backPropagation(error);
		}
	}

	//�t�`�d
	private void backPropagation2(float[][][] output){
		int layerSize = layerList.size();	//�w�̐�
		//�e�w�̏���ێ�
		CNNBaseLayer layer;
		float[][][] error = output;		//�`�d������덷
		
		//�ŏ��̑w�܂Ō덷��`�d
		for( int i = layerSize - 1 ; i >= 0 ; i--){
			layer = layerList.get(i);
			error = layer.backPropagation(error);
		}
	}
	
	//�O�w�̃p�����[�^���擾
	private int[] getBeforeLayerParameter(){
		if(layerList.size() == 0){
			return inputSize;
		}
		CNNBaseLayer layer = layerList.get(layerList.size()-1);
		return layer.getOutputSizeList();
	}
	
	//�p�����[�^�Z�b�g
	public void setParameter(int epochs , int trainDataNum , boolean useLearningDisplay){
		this.trainingEpochs = epochs;
		this.trainDataNum = trainDataNum;
		this.useLearningDisplay = useLearningDisplay;
	}
	
	//�f�[�^�Z�b�g����w�K
	public void startTraining(float[][][][] trainData , float[][] trainLabel){
		int data, epoch;
		//�w�K�󋵊m�F��ʂ𗘗p����ꍇ
		if(useLearningDisplay){
			float accuracy = 0.0f;	//�w�K�̐��x
			startLearningDisplay();	//�w�K�󋵊m�F��ʂ��N��
			for(epoch = 1 ; epoch <= trainingEpochs && learningDisplay.learningContinue ; epoch++){
				//�f�[�^����͂��w�K
				for(data = 0 ; data < trainDataNum && learningDisplay.learningContinue ; data++){
					training(trainData[data],trainLabel[data]);
					learningDisplay.updateTrainData(data);
				}
				//
				//�w�K�󋵊m�F��ʂ̍X�V
				accuracy = (float)correctNum / data;
				correctNum = 0;
				learningDisplay.updateParameter(epoch, learnRate, accuracy);	//�w�K�󋵊m�F��ʍX�V
			}
			learningDisplay.setStateText("�w�K���I�����܂����B");
		}
		//�w�K�󋵊m�F��ʂ𗘗p���Ȃ��ꍇ�i�p�t�H�[�}���X�D��j
		else{
			for(epoch = 1 ; epoch <= trainingEpochs ; epoch++){
				for(data = 0 ; data < trainDataNum ; data++){
					training(trainData[data],trainLabel[data]);
				}
			}
		}
	}
	
	//��p�`���w�K�f�[�^�Z�b�g����w�K���s��
	public void trainingByDataset(String fileName){
		try{
			// �w�K�p�f�[�^�ǂݍ���
			DataInputStream data = new DataInputStream(new FileInputStream(fileName));
			// �s���񐔓ǂݍ���
			int datanum = data.readInt();
			if(trainDataNum <= 0){
				trainDataNum = datanum;
			}
			if(784 != data.readInt()){
				data.close();
				throw new Exception("���͎��������قȂ��Ă��܂�");
			}
			if(10 != data.readInt()){
				data.close();
				throw new Exception("�o�͎��������قȂ��Ă��܂�");
			}
			// �̈�m��
			float[][][][] trainX = new float[trainDataNum][1][28][28]; // ���̓f�[�^
			float[][] trainY = new float[trainDataNum][10]; // ���t�f�[�^
			// ���̓f�[�^�ǂݍ���
			System.out.println("�t�@�C����[" + fileName + "] ���f�[�^��:" + trainDataNum+ " �ǂݍ��݊J�n");
			for (int numDataRead = 0 ; numDataRead < trainDataNum ; numDataRead++) {
				for (int i = 0; i < 784; i++) {
					trainX[numDataRead][0][i%28][i/28] = data.readUnsignedByte()==0?0:1;	//0�ȊO�͂��ׂ�1�Ƃ݂Ȃ�
				}if (numDataRead % 50 == 0) {
					System.out.print(".");
				}
				if ((numDataRead % 800) == 0) {
					System.out.println(" " + numDataRead + " / " + trainDataNum);
				}
			}
			System.out.println("");
			//���t�f�[�^�ǂݍ���
			for (int numDataRead = 0 ; numDataRead < trainDataNum ; numDataRead++) {
				setYClass(trainY,numDataRead,data.readInt());
			}
			//�ǂݍ��ݏI��
			data.close();
			//�w�K
			useLearningDisplay = true;
			startTraining(trainX,trainY);
			//�ۑ�
			if(isModelSave){
				save(modelFilePath);
			}
		}catch(IOException e){
			System.out.println("�G���[�F�t�@�C��["+ fileName +"]��������܂���");
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	//�N���X���Z�b�g
	private void setYClass(float[][] y , int setData , int setClass){
		for(int i = 0 ; i < 10 ; i++){
			y[setData][i] = (i==setClass)? 1 : 0;
		}
	}
	
	//���͌`���t�@�C������f�[�^����͂��N���X���ތ��ʂ�Ԃ�
	public int classifyByFile(String fileName){
		int output = getMaxIndex(calculateByFile(fileName));
		System.out.println("���̓t�@�C��["+ fileName +"] �o�͌���:"+output);
		return output;
	}
	
	//���͌`���t�@�C������f�[�^����͂��o�͂����߂�
	public float[][][] calculateByFile(String fileName){
		float[][][] input = new float[1][28][28];
		try {
			//�t�@�C�����̓X�g���[��
			BufferedReader br = new BufferedReader(new FileReader(fileName));
				
			//������ǂݍ���1���z��֑��
			int c = 0;	//�ǂݍ��񂾕������L��
			for(int i = 0 ; i < 784 ; i++){
				c = br.read();
				if(c == '0'){	//0�̏ꍇ
					input[0][i%28][i/28] = 0;
				}else if(c == '1'){	//1�̏ꍇ
					input[0][i%28][i/28] = 1;
				}else{	//�ǂݍ��߂Ȃ������̏ꍇ�͂�蒼��
					i--;
					continue;
				}
			}
			//�t�@�C���N���[�Y
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
		return propagation(input);
	}
	
	//�o�͂̐��l����ł��l���傫���v�f�̃C���f�b�N�X��Ԃ�
	private int getMaxIndex(final float[][][] output){
		int classLabel = 0;	//�ő�l���Ƃ�N���X
		float max = output[0][0][0];		//�ő�l
		//�o�͂��ő�ł���m�[�h�̔ԍ���Ԃ�
		for(int i = 1 ; i < 10 ; i++){
			if(max < output[i][0][0]){
				classLabel = i;
				max = output[i][0][0];
			}
		}
		return classLabel;
	} 
	
	//�~�j�o�b�`�w�K
	private float[][][][] totalInput;	//�e�w�ɂ��������
	public void minibatchTraining(float[][][][] trainData , float[][] trainLabel , int minibatchSize){
		startLearningDisplay();	//�w�K�󋵊m�F��ʂ��N��
		
		//�ϐ�
		float accuracy;
		int i, data;
		int minibatchIdx , trainStart , trainEnd;	//�~�j�o�b�`�ɂ��w�K�f�[�^�ԍ��̍ŏ��ƍŌ�
		boolean epochContinue;	//�~�j�o�b�`�ɂ��w�K�f�[�^���ꏄ�������ǂ���
		final int layerNum = layerList.size();
		final int outputSize = layerList.get(layerNum-1).sizeNout;
		float[][][] totalError = new float[outputSize][1][1];
		
		//�w�K�f�[�^���X�g�̍쐬
		ArrayList<Integer> trainList = new ArrayList<Integer>();	//�w�K���s������
		for(data = 0 ; data < trainDataNum ; data++){
			trainList.add(data);
		}
		
		//���񏈗��̏���
		ExecutorService pool = Executors.newFixedThreadPool(minibatchSize);
		AtomicInteger correctNum = new AtomicInteger();
		AtomicBoolean[] incorrectList = new AtomicBoolean[minibatchSize];
		AtomicFloat[] error = new AtomicFloat[outputSize];
		AtomicBoolean needTraining = new AtomicBoolean(false);	//�~�j�o�b�`����1�ł���F�����������ꍇ
		AtomicBoolean exceptionError = new AtomicBoolean(false);	//�o�͂ɃG���[������ꍇ
		totalInput = new float[layerList.size()][][][];
		for(i = 0; i < outputSize; i++){
			error[i] = new AtomicFloat();
		}
		for(i = 0; i < minibatchSize; i++){
			incorrectList[i] = new AtomicBoolean();
		}
		for(i = 0; i < layerList.size(); i++){
			layerList.get(i).initMinibatchSetting(minibatchSize);
		}
		
		//���ۂ̊w�K
		for(int epoch = 1 ; epoch <= trainingEpochs && learningDisplay.learningContinue; epoch++){
			//�~�j�o�b�`�w�K
			minibatchIdx = 0;
			epochContinue = true;
			Collections.shuffle(trainList);	//�w�K�f�[�^���͏���ύX
			while(epochContinue && learningDisplay.learningContinue){
				//�~�j�o�b�`�͈̔͂����߂�
				trainStart = minibatchIdx*minibatchSize;
				trainEnd = (minibatchIdx+1)*minibatchSize;
				if(trainEnd >= trainDataNum){
					trainEnd = trainDataNum;
					epochContinue = false;
				}
				//����̏���
				final CyclicBarrier barrier = new CyclicBarrier(trainEnd - trainStart + 1);
				needTraining.set(false);
				class CNNMiniBatchTask implements Runnable{
					int thread;
					float[][][] image;
					float[] label;
					
					//�^�X�N����
					public CNNMiniBatchTask(int thread, float[][][] image, float[] label){
						this.thread = thread;
						this.image = image;
						this.label = label;
					}

					//����
					public void run() {
						float[][][][] input = new float[layerNum+1][][][];
						input[0] = image;
						for(int i = 0; i < layerNum; i++){
							input[i+1] = layerList.get(i).propagation(input[i]);
						}
						if(Float.isNaN(input[layerNum][0][0][0])){
							exceptionError.set(true);
							/*for (int i = 0; i <= layerNum; i++) {
								for (int j = 0; j < input[i].length; j++) {
									for (int k = 0; k < input[i][j].length; k++) {
										System.out.println("THREAD:"+thread+" EXCEPTION INPUT "+i+"-"+j+"-"+k+":"+input[i][j][k][0]);
									}
								}
							}*/
						}
						/*if(check(input[layerNum],label)){
							incorrectList[thread].set(false);
							correctNum.incrementAndGet();
						}else*/{
							needTraining.set(true);
							incorrectList[thread].set(true);
							for(int i = 0; i < outputSize; i++){
								error[i].addAndGet(input[layerNum][i][0][0] - label[i]);
							}
							for(int i = 0; i < layerNum; i++){
								inputSet(i,input[i]);
							}
						}
						try {
							barrier.await();
						} catch (InterruptedException | BrokenBarrierException ex) {}
					}
					
				}
				// �S�f�[�^�𐳂����F�������ꍇ�͋t�`�d�����Ȃ�
				if (!exceptionError.get()) {
					// �f�[�^��������
					for (i = 0; i < outputSize; i++) {
						error[i].set(0.0f);
					}
					for (i = 0; i < layerNum; i++) {
						totalInput[i] = layerList.get(i).getInputClone();
					}
					// ���`�d�����s
					for (data = trainStart; data < trainEnd; data++) {
						pool.execute(new CNNMiniBatchTask(data - trainStart,
								trainData[trainList.get(data)],
								trainLabel[trainList.get(data)]));
					}
					// ������ҋ@
					try {
						barrier.await();
					} catch (InterruptedException | BrokenBarrierException ex) {
					}
					if(needTraining.get()){
					for (i = 0; i < outputSize; i++) {
						totalError[i][0][0] = error[i].get();
						//System.out.println(i+":"+totalError[i][0][0]);
					}
					/*
					 * for(data = trainStart; data < trainEnd; data++){ output =
					 * propagation(trainData[trainList.get(data)]);
					 * if(check(output,trainLabel[trainList.get(data)]))
					 * correctNum++; for(i = 0; i < output.length; i++){
					 * totalError[i][0][0] += (output[i][0][0] -
					 * trainLabel[trainList.get(data)][i]); } }
					 */
					// �w�K�ɗp������̓f�[�^��ݒ�
					for (i = 0; i < layerList.size(); i++) {
						layerList.get(i).setInput(totalInput[i]);
					}
					backPropagation2(totalError); // �덷�`�d�ɂ��w�K
					for (data = trainStart; data < trainEnd; data++) {
						if (incorrectList[data - trainStart].get()) {
							for (i = 0; i < layerList.size(); i++) {
								layerList.get(i).training();
							}
						}
					}
					}
				}
				learningDisplay.updateTrainData(trainEnd);
				minibatchIdx++;
				if(exceptionError.get()) {
					/*for (i = 0; i < layerList.size(); i++) {
						for (int j = 0; j < totalInput[i].length; j++) {
							for (int k = 0; k < totalInput[i][j].length; k++) {
								System.out.println("EXCEPTION INPUT "+i+"-"+j+"-"+k+":"+totalInput[i][j][k][0]);
							}
						}
					}*/
					learningDisplay.exceptionProcess();
				}
			}
			//�w�K�󋵊m�F��ʂ̍X�V
			accuracy = (float)correctNum.get() / data;
			correctNum.set(0);
			learningDisplay.updateParameter(epoch, learnRate, accuracy);	//�w�K�󋵊m�F��ʍX�V
		}
		totalInput = null;
		
		//���񏈗��I��
		pool.shutdown();
		
		learningDisplay.dispose();
		learningDisplay = null;
	}
	
	//�e�w�̓��͂̑��a���v�Z(�~�j�o�b�`�p)
	synchronized private void inputSet(int layerNumber, float[][][] input){
		int i, j, k;
		final int sizeN = input.length, sizeX = input[0].length, sizeY = input[0][0].length;
		for(i = 0; i < sizeN; i++){
			for(j = 0; j < sizeX; j++){
				for(k = 0; k < sizeY; k++){
					totalInput[layerNumber][i][j][k] += input[i][j][k];
				}
			}
		}
	}
	
	//�w�K�f�[�^1�ɑ΂���w�K
	public void training(float[][][] image , float[] label){
		float[][][] output = propagation(image);			//���`�d
		if(useLearningDisplay){
			if(check(output,label)) correctNum++;
		}
		backPropagation(output , label);	//�t�`�d
	}
	
	//CNN�ɂ��f�[�^�����ʂ����x���ƈ�v�����ꍇ��true��Ԃ�
	private boolean check(float[][][] output , float[] label){
		int size = output.length;
		float max = output[0][0][0];
		int maxIndex = 0;
		int labelIndex = 0;
		for(int i = 1 ; i < size ; i++){
			if(max < output[i][0][0]){
				max = output[i][0][0];
				maxIndex = i;
			}
			if(label[i] > 0.5f) labelIndex = i;
		}
		return maxIndex == labelIndex;
	}
	
	//�摜�̐��K��
	public float[][][] normalization(float[][][] input){
		int c , x , y;
		int sizeC = input.length;
		int sizeX = input[0].length;
		int sizeY = input[0][0].length;
		float[][][] output = new float[sizeC][sizeX][sizeY];
		float[] mean = new float[sizeC];	//����
		float[] var = new float[sizeC];		//���U
		float sum;	//���v�l
		//���ς��Z�o
		for(c = 0 ; c < sizeC ; c++){
			sum = 0;
			for(x = 0 ; x < sizeX ; x++){
				for(y = 0 ; y < sizeY ; y++){
					sum += input[c][x][y];
				}
			}
			mean[c] = sum / (sizeX*sizeY);
			//System.out.println("MEAN"+c+":"+mean[c]);
		}
		//���U���Z�o
		for(c = 0 ; c < sizeC ; c++){
			sum = 0;
			for(x = 0 ; x < sizeX ; x++){
				for(y = 0 ; y < sizeY ; y++){
					sum += input[c][x][y]*input[c][x][y];
				}
			}
			var[c] = sum/(sizeX*sizeY) - mean[c]*mean[c];
			//System.out.println("VAR"+c+":"+var[c]);
		}
		//�o�͂��Z�o
		for(c = 0 ; c < sizeC ; c++){
			for(x = 0 ; x < sizeX ; x++){
				for(y = 0 ; y < sizeY ; y++){
					output[c][x][y] = (input[c][x][y] - mean[c]) / var[c];
				}
			} 
		}
		return output;
	}
	
	//����
	public float[] classify(float[][][] input){
		float[][][] output1 = propagation(input);
		int size = output1.length;
		float[] output2 = new float[size];
		for(int i = 0 ; i < size ; i++){
			output2[i] = output1[i][0][0];
		}
		return output2;
	}
	
	//�e�X�g�p
	public static void main(String[] args){
		CNN cnn = new CNN();
		
		float[][][][] train = {
				{{	{1.0f , 1.0f , 1.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 1.0f , 1.0f , 0.0f},
					{0.0f , 1.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 1.0f , 0.0f},
					{1.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 0.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 1.0f}}},
				
				{{	{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 1.0f},
					{0.0f , 1.0f , 0.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 1.0f}}}
		};
		
		float[][] label = {	{1.0f , 0.0f , 0.0f},
							{1.0f , 0.0f , 0.0f},
							{0.0f , 1.0f , 0.0f},
							{0.0f , 1.0f , 0.0f},
							{0.0f , 0.0f , 1.0f},
							{0.0f , 0.0f , 1.0f}};
		
		float[][][][] test = {
				{{	{1.0f , 1.0f , 1.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 1.0f , 0.0f},
					{1.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 0.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 1.0f}}},
				
				{{	{0.0f , 1.0f , 1.0f , 0.0f , 0.0f},
					{1.0f , 1.0f , 0.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 0.0f , 0.0f , 1.0f},
					{0.0f , 0.0f , 0.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 0.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 1.0f},
					{0.0f , 1.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{1.0f , 1.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
				
				{{	{0.0f , 0.0f , 1.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 0.0f , 0.0f , 0.0f},
					{1.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f}}},
					
				{{	{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 1.0f , 0.0f , 1.0f},
					{0.0f , 0.0f , 1.0f , 1.0f , 1.0f}}}
		};
		
		//�w�K��
		cnn.addConvolutionLayer(1, 5, 5, 2, 5);
		cnn.addConvolutionLayer(2,4);
		cnn.addConvolutionLayer(2,2);
		cnn.addFullConnectLayer(3);
		
		//�w�K
		for(int epoch = 0 ; epoch < 20000 ; epoch++){
			for(int data = 0 ; data < 6 ; data++){
				//System.out.println("EPOCH:"+(epoch+1)+" DATA:"+data);
				cnn.training(train[data], label[data]);
			}
		}
		
		//�e�X�g
		float[][][] output = new float[3][1][1];
		for(int data = 0 ; data < test.length ; data++){
			System.out.println("TEST DATA:"+data);
			output = cnn.propagation(test[data]);
			for(int i = 0 ; i < 3 ; i++){
				System.out.print(output[i][0][0] + " ");
			}
			System.out.println();
		}
	}
}