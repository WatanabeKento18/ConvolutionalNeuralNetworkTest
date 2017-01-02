package deeplearning;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * 
 *	畳み込みニューラルネットワーク
 *	CNNConvolution,CNNActivation,CNNPooling,CNNFullConnectより構成
 *
**/

public class CNN{
	private ArrayList<CNNBaseLayer> layerList;	//CNNを構成する層のリスト
	private int convLayerNum;				//畳み込み層の数
	private int count;

	//学習状況確認画面に関する設定
	private boolean useLearningDisplay;		//学習状況確認画面を利用するかどうか
	private LearningDisplay learningDisplay;	//学習状況確認画面
	private int correctNum;					//学習精度調査用
	
	//学習モデル保存パラメータ
	private boolean isModelSave;
	private String modelFilePath;
	
	//学習用に保持しておくパラメータ
	private int[] inputSize;	//入力層サイズ
	private int trainingEpochs;	//学習回数
	private int trainDataNum;	//学習データ数
	private float learnRate;		//学習率
	
	enum LayerCode{
		CONVOLUTION , ACTIVATION , POOLING , FULLCONNECT
	};
	
	public CNN(){
		layerList = new ArrayList<CNNBaseLayer>();
		convLayerNum = 0; 
	}
	
	//CNNの状態をリセットする
	public void reset(){
		layerList.clear();
		convLayerNum = 0;
	}
	
	//学習モデル保存に関する設定
	public void setModelSave(String modelFilePath){
		if(modelFilePath == null){
			isModelSave = false;
			return;
		}
		this.modelFilePath = modelFilePath;
		isModelSave = true;
	}
	
	//学習モデルを保存する
	public void save(String fileName) {
		int layerSize = layerList.size();
		CNNBaseLayer layer;
		try {
			// ファイル出力ストリーム
			BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
			bw.write(String.format("%d", layerList.size()));	//層の数
			bw.newLine();
			
			//各層の情報を書き込み
			for(int i = 0 ; i < layerSize ; i++){
				layer = layerList.get(i);
				layer.save(bw);
			}
			// ファイルクローズ
			bw.close();
			System.out.println("学習モデルをファイル[" + fileName + "]で保存しました");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//学習モデルからCNNを構築する
	public void load(String fileName){
		reset();
		int sizeNin , sizeNout , sizeX , sizeY , sizeF , colorNum;
		CNNBaseLayer layer;
		try {
			//ファイル入力ストリーム
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			int layerSize = Integer.parseInt(br.readLine());	//層の数
			
			//各層を順番に読み込み
			for(int i = 0 ; i < layerSize ; i++){
				//層の識別番号を確認
				int layerType = Integer.parseInt(br.readLine());	//層の数
				if(layerType < 1000){
					System.out.println("学習モデル読み込みエラー");
					return;
				}
				
				//層ごとに読み込み
				switch(layerType){
				//畳み込み層
				case 1011:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					sizeF = Integer.parseInt(br.readLine());
					sizeNout = Integer.parseInt(br.readLine());
					learnRate = Float.parseFloat(br.readLine());
					addConvolutionLayer(sizeNin , sizeX , sizeY , sizeF , sizeNout);	//新しく層を登録
					layer = layerList.get(layerList.size()-1);
					layer.load(br);
					break;
				//活性化層
				case 1021:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					addActivationLayer(sizeNin , sizeX , sizeY);	//新しく層を登録
					break;
				//プーリング層
				case 1031:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					addPoolingLayer(sizeNin , sizeX , sizeY);	//新しく層を登録
					break;
				//全結合層
				case 1041:
					sizeNin = Integer.parseInt(br.readLine());
					sizeX = Integer.parseInt(br.readLine());
					sizeY = Integer.parseInt(br.readLine());
					sizeNout = Integer.parseInt(br.readLine());
					learnRate = Float.parseFloat(br.readLine());
					addFullConnectLayer(sizeNin , sizeX , sizeY , sizeNout);	//新しく層を登録
					layer = layerList.get(layerList.size()-1);
					layer.load(br);
					break;
				}
			}
			//ファイルクローズ
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//学習状況確認画面を起動
	public void startLearningDisplay(){
		learningDisplay = new LearningDisplay();
		learningDisplay.setup();
		learningDisplay.setParameter(trainingEpochs, trainDataNum);
		learningDisplay.startTraining();
	}
	
	//入力データサイズを定義
	public void setInputSize(int color, int x, int y){
		inputSize = new int[3];
		inputSize[0] = color;
		inputSize[1] = x;
		inputSize[2] = y;
	}
	
	//畳み込み層を追加
	//(nodeNum:入力ノード数 inputSizeX:入力データXサイズ inputSizeY:入力データYサイズ filterSize:フィルタ幅 kernelNum:フィルタの数)
	public void addConvolutionLayer(int nodeNum , int inputSizeX , int inputSizeY , int filterSize , int kernelNum ){
		boolean isTop = false;
		if(convLayerNum == 0)	isTop = true;	//畳み込み層の中で先頭の場合
		layerList.add(new CNNConvolution(nodeNum,inputSizeX,inputSizeY,filterSize,kernelNum,isTop));
		convLayerNum++;
	}
	//パラメータを自動で判別して畳み込み層を追加
	//(filterSize:フィルタ幅 kernelNum:フィルタの数)
	public void addConvolutionLayer(int filterSize , int kernelNum){
		boolean isTop = false;
		if(convLayerNum == 0)	isTop = true;	//畳み込み層の中で先頭の場合
		int[] parameter = getBeforeLayerParameter();
		layerList.add(new CNNConvolution(parameter[0],parameter[1],parameter[2],filterSize,kernelNum,isTop));
		convLayerNum++;
	}
	
	//活性化層を追加
	//(nodeNum:入力ノード数 inputSizeX:入力データXサイズ inputSizeY:入力データYサイズ filterSize:フィルタ幅 kernelNum:フィルタの数)
	public void addActivationLayer(int nodeNum , int inputSizeX , int inputSizeY){
		layerList.add(new CNNActivation(nodeNum,inputSizeX,inputSizeY));
	}
	//パラメータを自動で判別して活性化層を追加
	public void addActivationLayer(){
		addLayer(1021);
	}
	
	//プーリング層を追加
	//(nodeNum:入力ノード数 inputSizeX:入力データXサイズ inputSizeY:入力データYサイズ filterSize:フィルタ幅 kernelNum:フィルタの数)
	public void addPoolingLayer(int nodeNum , int inputSizeX , int inputSizeY){
		layerList.add(new CNNPooling(nodeNum,inputSizeX,inputSizeY));
	}
	//パラメータを自動で判別してプーリング層を追加
	public void addPoolingLayer(){
		addLayer(1031);
	}
	
	//全結合層1を追加
	//(nodeNum:入力ノード数 inputSizeX:入力データXサイズ inputSizeY:入力データYサイズ outputSize:出力サイズ)
	public void addFullConnectLayer(int nodeNum , int inputSizeX , int inputSizeY , int outputSize){
		layerList.add(new CNNFullConnectSoftMax(nodeNum,inputSizeX,inputSizeY,outputSize));
	}
	//パラメータを自動で判別して全結合層を追加
	//(outputSize:出力サイズ)
	public void addFullConnectLayer(int outputSize){
		addLayer(1041, outputSize);
	}
	
	//コードを指定して層を追加
	public void addLayer(int code, int parameter1, int parameter2){
		int[] parameter = getBeforeLayerParameter();
		switch(code){
		//畳み込み層 パラメータ1:フィルタ数 パラメータ2:フィルタサイズ
		case 1011:
			layerList.add(new CNNConvolution(parameter[0],parameter[1],parameter[2],parameter2,parameter1,true));
			break;
		//活性化層
		case 1021:
			layerList.add(new CNNActivation(parameter[0],parameter[1],parameter[2]));
			break;
		//プーリング層
		case 1031:
			layerList.add(new CNNPooling(parameter[0],parameter[1],parameter[2]));
			break;
		//全結合層 パラメータ1:出力次元数
		case 1041:	//全結合+SoftMax層
			layerList.add(new CNNFullConnectSoftMax(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		case 1042:	//全結合+ReLU層
			layerList.add(new CNNFullConnectReLU(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		case 1043:	//全結合活性化無し
			layerList.add(new CNNFullConnectNone(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		//正規化層 パラメータ1:フィルタサイズ
		case 1051:	//局所コントラスト正規化
			layerList.add(new CNNNormalization(parameter[0],parameter[1],parameter[2],parameter1));
		break;
		case 1052:	//減算正規化
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
	
	//順伝播
	public float[][][] propagation(float[][][] input){
		int layerSize = layerList.size();	//層の数
		CNNBaseLayer layer;
		float[][][] output = input;
		/*float[] output2 = null;
		boolean convolutionLayerEnd = false;	//全結合層まで伝播したらtrue*/
		
		//最初の層から全結合層までを伝播
		for(int i = 0 ; i < layerSize ; i++){
			layer = layerList.get(i);
			output = layer.propagation(output);
		}
		
		//全結合層から最後の層まで伝播
		/*for(; layer < layerSize ; layer++){
			layerCode = layerList.get(layer);
			layerType = layerCode/100;
			layerNumber = layerCode%100;
			
			if(layerType != 5) return null;	//全結合層2ではない場合は不可
			CNNFullConnect2 fc = fc2Layer.get(layerNumber);
			output2 = fc.propagation(output2);
		}
		return output2;*/
		return output;
	}

	//逆伝播
	private void backPropagation(float[][][] output , float[] label){
		int layerSize = layerList.size();	//層の数
		//各層の情報を保持
		int i = layerSize-1;	//参照する層
		CNNBaseLayer layer;
		float[][][] error = null;		//伝播させる誤差
		
		//最終層が全結合層になっているかを確認し教師信号から誤差を算出
		layer = layerList.get(i);
		if(layer instanceof CNNBaseLayerFullConnect){
			error = ((CNNBaseLayerFullConnect)layer).backPropagation(output , label);
		}
		
		//全結合層から最初の層まで誤差を伝播
		for( i = layerSize - 2 ; i >= 0 ; i--){
			layer = layerList.get(i);
			error = layer.backPropagation(error);
		}
	}

	//逆伝播
	private void backPropagation2(float[][][] output){
		int layerSize = layerList.size();	//層の数
		//各層の情報を保持
		CNNBaseLayer layer;
		float[][][] error = output;		//伝播させる誤差
		
		//最初の層まで誤差を伝播
		for( int i = layerSize - 1 ; i >= 0 ; i--){
			layer = layerList.get(i);
			error = layer.backPropagation(error);
		}
	}
	
	//前層のパラメータを取得
	private int[] getBeforeLayerParameter(){
		if(layerList.size() == 0){
			return inputSize;
		}
		CNNBaseLayer layer = layerList.get(layerList.size()-1);
		return layer.getOutputSizeList();
	}
	
	//パラメータセット
	public void setParameter(int epochs , int trainDataNum , boolean useLearningDisplay){
		this.trainingEpochs = epochs;
		this.trainDataNum = trainDataNum;
		this.useLearningDisplay = useLearningDisplay;
	}
	
	//データセットから学習
	public void startTraining(float[][][][] trainData , float[][] trainLabel){
		int data, epoch;
		//学習状況確認画面を利用する場合
		if(useLearningDisplay){
			float accuracy = 0.0f;	//学習の精度
			startLearningDisplay();	//学習状況確認画面を起動
			for(epoch = 1 ; epoch <= trainingEpochs && learningDisplay.learningContinue ; epoch++){
				//データを入力し学習
				for(data = 0 ; data < trainDataNum && learningDisplay.learningContinue ; data++){
					training(trainData[data],trainLabel[data]);
					learningDisplay.updateTrainData(data);
				}
				//
				//学習状況確認画面の更新
				accuracy = (float)correctNum / data;
				correctNum = 0;
				learningDisplay.updateParameter(epoch, learnRate, accuracy);	//学習状況確認画面更新
			}
			learningDisplay.setStateText("学習を終了しました。");
		}
		//学習状況確認画面を利用しない場合（パフォーマンス優先）
		else{
			for(epoch = 1 ; epoch <= trainingEpochs ; epoch++){
				for(data = 0 ; data < trainDataNum ; data++){
					training(trainData[data],trainLabel[data]);
				}
			}
		}
	}
	
	//専用形式学習データセットから学習を行う
	public void trainingByDataset(String fileName){
		try{
			// 学習用データ読み込み
			DataInputStream data = new DataInputStream(new FileInputStream(fileName));
			// 行数列数読み込み
			int datanum = data.readInt();
			if(trainDataNum <= 0){
				trainDataNum = datanum;
			}
			if(784 != data.readInt()){
				data.close();
				throw new Exception("入力次元数が異なっています");
			}
			if(10 != data.readInt()){
				data.close();
				throw new Exception("出力次元数が異なっています");
			}
			// 領域確保
			float[][][][] trainX = new float[trainDataNum][1][28][28]; // 入力データ
			float[][] trainY = new float[trainDataNum][10]; // 教師データ
			// 入力データ読み込み
			System.out.println("ファイル名[" + fileName + "] 総データ数:" + trainDataNum+ " 読み込み開始");
			for (int numDataRead = 0 ; numDataRead < trainDataNum ; numDataRead++) {
				for (int i = 0; i < 784; i++) {
					trainX[numDataRead][0][i%28][i/28] = data.readUnsignedByte()==0?0:1;	//0以外はすべて1とみなす
				}if (numDataRead % 50 == 0) {
					System.out.print(".");
				}
				if ((numDataRead % 800) == 0) {
					System.out.println(" " + numDataRead + " / " + trainDataNum);
				}
			}
			System.out.println("");
			//教師データ読み込み
			for (int numDataRead = 0 ; numDataRead < trainDataNum ; numDataRead++) {
				setYClass(trainY,numDataRead,data.readInt());
			}
			//読み込み終了
			data.close();
			//学習
			useLearningDisplay = true;
			startTraining(trainX,trainY);
			//保存
			if(isModelSave){
				save(modelFilePath);
			}
		}catch(IOException e){
			System.out.println("エラー：ファイル["+ fileName +"]が見つかりません");
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	//クラスをセット
	private void setYClass(float[][] y , int setData , int setClass){
		for(int i = 0 ; i < 10 ; i++){
			y[setData][i] = (i==setClass)? 1 : 0;
		}
	}
	
	//入力形式ファイルからデータを入力しクラス分類結果を返す
	public int classifyByFile(String fileName){
		int output = getMaxIndex(calculateByFile(fileName));
		System.out.println("入力ファイル["+ fileName +"] 出力結果:"+output);
		return output;
	}
	
	//入力形式ファイルからデータを入力し出力を求める
	public float[][][] calculateByFile(String fileName){
		float[][][] input = new float[1][28][28];
		try {
			//ファイル入力ストリーム
			BufferedReader br = new BufferedReader(new FileReader(fileName));
				
			//数字を読み込み1つずつ配列へ代入
			int c = 0;	//読み込んだ文字を記憶
			for(int i = 0 ; i < 784 ; i++){
				c = br.read();
				if(c == '0'){	//0の場合
					input[0][i%28][i/28] = 0;
				}else if(c == '1'){	//1の場合
					input[0][i%28][i/28] = 1;
				}else{	//読み込めない文字の場合はやり直し
					i--;
					continue;
				}
			}
			//ファイルクローズ
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
		return propagation(input);
	}
	
	//出力の数値から最も値が大きい要素のインデックスを返す
	private int getMaxIndex(final float[][][] output){
		int classLabel = 0;	//最大値をとるクラス
		float max = output[0][0][0];		//最大値
		//出力が最大であるノードの番号を返す
		for(int i = 1 ; i < 10 ; i++){
			if(max < output[i][0][0]){
				classLabel = i;
				max = output[i][0][0];
			}
		}
		return classLabel;
	} 
	
	//ミニバッチ学習
	private float[][][][] totalInput;	//各層における入力
	public void minibatchTraining(float[][][][] trainData , float[][] trainLabel , int minibatchSize){
		startLearningDisplay();	//学習状況確認画面を起動
		
		//変数
		float accuracy;
		int i, data;
		int minibatchIdx , trainStart , trainEnd;	//ミニバッチによる学習データ番号の最初と最後
		boolean epochContinue;	//ミニバッチにより学習データを一巡したかどうか
		final int layerNum = layerList.size();
		final int outputSize = layerList.get(layerNum-1).sizeNout;
		float[][][] totalError = new float[outputSize][1][1];
		
		//学習データリストの作成
		ArrayList<Integer> trainList = new ArrayList<Integer>();	//学習を行う順番
		for(data = 0 ; data < trainDataNum ; data++){
			trainList.add(data);
		}
		
		//並列処理の準備
		ExecutorService pool = Executors.newFixedThreadPool(minibatchSize);
		AtomicInteger correctNum = new AtomicInteger();
		AtomicBoolean[] incorrectList = new AtomicBoolean[minibatchSize];
		AtomicFloat[] error = new AtomicFloat[outputSize];
		AtomicBoolean needTraining = new AtomicBoolean(false);	//ミニバッチ内に1つでも誤認識があった場合
		AtomicBoolean exceptionError = new AtomicBoolean(false);	//出力にエラーがある場合
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
		
		//実際の学習
		for(int epoch = 1 ; epoch <= trainingEpochs && learningDisplay.learningContinue; epoch++){
			//ミニバッチ学習
			minibatchIdx = 0;
			epochContinue = true;
			Collections.shuffle(trainList);	//学習データ入力順を変更
			while(epochContinue && learningDisplay.learningContinue){
				//ミニバッチの範囲を求める
				trainStart = minibatchIdx*minibatchSize;
				trainEnd = (minibatchIdx+1)*minibatchSize;
				if(trainEnd >= trainDataNum){
					trainEnd = trainDataNum;
					epochContinue = false;
				}
				//並列の準備
				final CyclicBarrier barrier = new CyclicBarrier(trainEnd - trainStart + 1);
				needTraining.set(false);
				class CNNMiniBatchTask implements Runnable{
					int thread;
					float[][][] image;
					float[] label;
					
					//タスク生成
					public CNNMiniBatchTask(int thread, float[][][] image, float[] label){
						this.thread = thread;
						this.image = image;
						this.label = label;
					}

					//処理
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
				// 全データを正しく認識した場合は逆伝播をしない
				if (!exceptionError.get()) {
					// データを初期化
					for (i = 0; i < outputSize; i++) {
						error[i].set(0.0f);
					}
					for (i = 0; i < layerNum; i++) {
						totalInput[i] = layerList.get(i).getInputClone();
					}
					// 順伝播を実行
					for (data = trainStart; data < trainEnd; data++) {
						pool.execute(new CNNMiniBatchTask(data - trainStart,
								trainData[trainList.get(data)],
								trainLabel[trainList.get(data)]));
					}
					// 処理を待機
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
					// 学習に用いる入力データを設定
					for (i = 0; i < layerList.size(); i++) {
						layerList.get(i).setInput(totalInput[i]);
					}
					backPropagation2(totalError); // 誤差伝播により学習
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
			//学習状況確認画面の更新
			accuracy = (float)correctNum.get() / data;
			correctNum.set(0);
			learningDisplay.updateParameter(epoch, learnRate, accuracy);	//学習状況確認画面更新
		}
		totalInput = null;
		
		//並列処理終了
		pool.shutdown();
		
		learningDisplay.dispose();
		learningDisplay = null;
	}
	
	//各層の入力の総和を計算(ミニバッチ用)
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
	
	//学習データ1つに対する学習
	public void training(float[][][] image , float[] label){
		float[][][] output = propagation(image);			//順伝播
		if(useLearningDisplay){
			if(check(output,label)) correctNum++;
		}
		backPropagation(output , label);	//逆伝播
	}
	
	//CNNによりデータを識別しラベルと一致した場合にtrueを返す
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
	
	//画像の正規化
	public float[][][] normalization(float[][][] input){
		int c , x , y;
		int sizeC = input.length;
		int sizeX = input[0].length;
		int sizeY = input[0][0].length;
		float[][][] output = new float[sizeC][sizeX][sizeY];
		float[] mean = new float[sizeC];	//平均
		float[] var = new float[sizeC];		//分散
		float sum;	//合計値
		//平均を算出
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
		//分散を算出
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
		//出力を算出
		for(c = 0 ; c < sizeC ; c++){
			for(x = 0 ; x < sizeX ; x++){
				for(y = 0 ; y < sizeY ; y++){
					output[c][x][y] = (input[c][x][y] - mean[c]) / var[c];
				}
			} 
		}
		return output;
	}
	
	//識別
	public float[] classify(float[][][] input){
		float[][][] output1 = propagation(input);
		int size = output1.length;
		float[] output2 = new float[size];
		for(int i = 0 ; i < size ; i++){
			output2[i] = output1[i][0][0];
		}
		return output2;
	}
	
	//テスト用
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
		
		//学習器
		cnn.addConvolutionLayer(1, 5, 5, 2, 5);
		cnn.addConvolutionLayer(2,4);
		cnn.addConvolutionLayer(2,2);
		cnn.addFullConnectLayer(3);
		
		//学習
		for(int epoch = 0 ; epoch < 20000 ; epoch++){
			for(int data = 0 ; data < 6 ; data++){
				//System.out.println("EPOCH:"+(epoch+1)+" DATA:"+data);
				cnn.training(train[data], label[data]);
			}
		}
		
		//テスト
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