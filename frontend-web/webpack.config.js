
const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const { VueLoaderPlugin } = require('vue-loader/dist/index')
 
module.exports = {
  mode: 'development', // 环境模式
  entry: path.resolve(__dirname, './src/main.js'), // 打包入口
  output: {
    path: path.resolve(__dirname, 'dist'), // 打包出口
    filename: 'js/[name].js' // 打包完的静态资源文件名
  },
  //OpenCV.js
  //Test.js
  module: {
    rules: [
      {
        test: /\.vue$/,
        use: [
          'vue-loader'
        ]
      },
      {
        test: /\.m?js$/,
        exclude: /(node_modules|bower_components)/,
        loader: 'babel-loader',
        options: {
          presets: ["@babel/preset-env"]
        }
        
      },
      {
        test: /\.css$/,
          use: [
            'vue-style-loader',
            'css-loader'
          ]
      },
      {
        test: /\.sass$/,
        use: [
          'vue-style-loader',
          'css-loader',
          {
            loader: 'sass-loader',
            options: {
              indentedSyntax: true,
              // sass-loader version >= 8
              sassOptions: {
                indentedSyntax: true
              }
            }
          }
        ]
      },
      {
        test: /\.scss$/,
        use: [
          'vue-style-loader',
          'css-loader',
          'sass-loader'
        ]
      }
    ]
  },
  resolve: {
    fallback: {
      fs: false,
      path:false,
      crypto:false
    }
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: path.resolve(__dirname, './index.html'), // 我们要使用的 html 模板地址
      filename: 'index.html',
      title: 'Hello'
    }),
    new VueLoaderPlugin()
  ]

}