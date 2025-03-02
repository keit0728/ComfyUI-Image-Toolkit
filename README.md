# ComfyUI-Image-Toolkit

**ComfyUI 用カスタムノードパック**

このノードパックは、画像処理と変換のための便利なツールを提供します。

## 機能

このパックには以下のノードが含まれています：

### BrightnessTransparency

明るさに基づいて透明度を調整します。明るい部分は透明に、暗い部分は不透明になります。

### BinarizeImage

画像を 2 値化します。指定した閾値より明るい部分は白（1.0）に、暗い部分は黒（0.0）に変換されます。

### GrayscaleImage

カラー画像をグレースケールに変換します。RGB 値を均等に処理して、グレースケール画像を生成します。

### AntialiasingImage

アンチエイリアシング効果を画像に適用します。画像を一度拡大してから元のサイズに戻すことで、エッジを滑らかにします。

## インストール方法

### 推奨方法

- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)を使用してインストールします。

### 手動インストール

1. `ComfyUI/custom_nodes`ディレクトリに移動します。
2. 以下のコマンドでリポジトリをクローンします：
   ```
   git clone https://github.com/your-username/ComfyUI-Image-Toolkit
   cd ComfyUI-Image-Toolkit
   ```

## 使用方法

1. ComfyUI を起動します。
2. ノードブラウザで「ComfyUI-Image-Toolkit」カテゴリを探します。
3. 必要なノードをワークフローに追加して使用します。

## ライセンス

GPL-3.0 license
