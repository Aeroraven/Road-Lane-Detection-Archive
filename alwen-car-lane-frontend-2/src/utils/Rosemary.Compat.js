"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _regenerator = _interopRequireDefault(require("@babel/runtime/regenerator"));

var _asyncToGenerator2 = _interopRequireDefault(require("@babel/runtime/helpers/asyncToGenerator"));

var _classCallCheck2 = _interopRequireDefault(require("@babel/runtime/helpers/classCallCheck"));

var _createClass2 = _interopRequireDefault(require("@babel/runtime/helpers/createClass"));

var _defineProperty2 = _interopRequireDefault(require("@babel/runtime/helpers/defineProperty"));

var _axios = _interopRequireDefault(require("axios"));

var _engine = require("engine.io-client");

var _sass = require("sass");

var _socket = _interopRequireDefault(require("socket.io-client"));

var Rosemary = /*#__PURE__*/function () {
  function Rosemary() {
    (0, _classCallCheck2["default"])(this, Rosemary);
    (0, _defineProperty2["default"])(this, "serverLists", {});
  }

  (0, _createClass2["default"])(Rosemary, [{
    key: "getChromeVer",
    value: function getChromeVer() {
      var ua = navigator.userAgent;
      var b = ua.indexOf("Chrome/");

      if (b < 0) {
        return 0;
      }

      return parseFloat(ua.substring(b + 8, ua.lastIndexOf("\.")));
    }
  }, {
    key: "getBroswer",
    value: function getBroswer() {
      var sys = {};
      var ua = navigator.userAgent.toLowerCase();
      var s;
      (s = ua.match(/edge\/([\d.]+)/)) ? sys.edge = s[1] : (s = ua.match(/rv:([\d.]+)\) like gecko/)) ? sys.ie = s[1] : (s = ua.match(/msie ([\d.]+)/)) ? sys.ie = s[1] : (s = ua.match(/firefox\/([\d.]+)/)) ? sys.firefox = s[1] : (s = ua.match(/chrome\/([\d.]+)/)) ? sys.chrome = s[1] : (s = ua.match(/opera.([\d.]+)/)) ? sys.opera = s[1] : (s = ua.match(/version\/([\d.]+).*safari/)) ? sys.safari = s[1] : 0;
      if (sys.edge) return {
        broswer: "Edge",
        version: sys.edge
      };
      if (sys.ie) return {
        broswer: "IE",
        version: sys.ie
      };
      if (sys.firefox) return {
        broswer: "Firefox",
        version: sys.firefox
      };
      if (sys.chrome) return {
        broswer: "Chrome",
        version: sys.chrome
      };
      if (sys.opera) return {
        broswer: "Opera",
        version: sys.opera
      };
      if (sys.safari) return {
        broswer: "Safari",
        version: sys.safari
      };
      return {
        broswer: "",
        version: "0"
      };
    }
  }, {
    key: "rosLoader",
    value: function () {
      var _rosLoader = (0, _asyncToGenerator2["default"])( /*#__PURE__*/_regenerator["default"].mark(function _callee(successHandler, progressHandler) {
        var errorHandler,
            ua,
            info,
            infos,
            steps,
            i,
            _args = arguments;
        return _regenerator["default"].wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                errorHandler = _args.length > 2 && _args[2] !== undefined ? _args[2] : function (x) {
                  window.alert("发生错误", x);
                };
                ua = navigator.userAgent.toLowerCase();
                progressHandler("运行环境检查");
                info = {
                  ie: /msie/.test(ua) && !/opera/.test(ua),
                  op: /opera/.test(ua),
                  sa: /version.*safari/.test(ua),
                  ch: /chrome/.test(ua),
                  ff: /gecko/.test(ua) && !/webkit/.test(ua)
                };
                infos = this.getBroswer();

                if (!(info.sa == false && info.ch == false && info.ff == false)) {
                  _context.next = 9;
                  break;
                }

                progressHandler("当前浏览器或系统配置不受支持,请更换为Chrome,FireFox或Safari内核的浏览器");
                alert("浏览器或系统配置不受支持");
                return _context.abrupt("return");

              case 9:
                console.log("A");

                if (!(info.ch == true || info.ff == true)) {
                  _context.next = 15;
                  break;
                }

                if (!(parseInt(infos.version) < 86)) {
                  _context.next = 15;
                  break;
                }

                progressHandler("浏览器内核或WebView版本(" + parseInt(infos.version) + ")过低，请升级至Chrome/Firefox或WebView>=89以上版本");
                alert("您使用的浏览器内核版本无法支持应用运行。由于使用到新的HTML5特性和ES7语法，应用最低支持的内核版本为89。" + "对于移动端用户，请在调整网络配置后，在Google Play应用商店中升级系统WebView版本" + "(https://play.google.com/store/apps/details?id=com.google.android.webview)。");
                return _context.abrupt("return");

              case 15:
                if (!(info.sa == true)) {
                  _context.next = 20;
                  break;
                }

                if (!(parseInt(infos.version) < 15)) {
                  _context.next = 20;
                  break;
                }

                progressHandler("浏览器内核版本过低，请升级至Safari>=15以上版本");
                alert("您使用的浏览器内核版本无法支持应用运行。由于使用到新的HTML5特性和ES7语法，应用最低支持的内核版本为15。" + "对于iOS用户，如果您需要运行本应用，您需要在“设置-General-SoftwareUpdate”升级iOS版本至>=15");
                return _context.abrupt("return");

              case 20:
                successHandler();
                steps = [["加载字体", this.rosFontLoading], ["查询可用服务器列表", this.rosServerListing], ["测试服务器状态", this.rosServerChecking], ["测试Socket状态", this.rosServerCheckingSocket], ["初始化完成", this.rosDone]];
                _context.prev = 22;
                i = 0;

              case 24:
                if (!(i < steps.length)) {
                  _context.next = 32;
                  break;
                }

                progressHandler(steps[i][0]);
                _context.next = 28;
                return steps[i][1](successHandler, progressHandler, errorHandler);

              case 28:
                successHandler();

              case 29:
                i++;
                _context.next = 24;
                break;

              case 32:
                _context.next = 37;
                break;

              case 34:
                _context.prev = 34;
                _context.t0 = _context["catch"](22);
                errorHandler(_context.t0);

              case 37:
              case "end":
                return _context.stop();
            }
          }
        }, _callee, this, [[22, 34]]);
      }));

      function rosLoader(_x, _x2) {
        return _rosLoader.apply(this, arguments);
      }

      return rosLoader;
    }()
  }, {
    key: "rosDone",
    value: function () {
      var _rosDone = (0, _asyncToGenerator2["default"])( /*#__PURE__*/_regenerator["default"].mark(function _callee2(successHandler, progressHandler, errorHandler) {
        return _regenerator["default"].wrap(function _callee2$(_context2) {
          while (1) {
            switch (_context2.prev = _context2.next) {
              case 0:
              case "end":
                return _context2.stop();
            }
          }
        }, _callee2);
      }));

      function rosDone(_x3, _x4, _x5) {
        return _rosDone.apply(this, arguments);
      }

      return rosDone;
    }()
  }, {
    key: "rosServerCheckingSocket",
    value: function () {
      var _rosServerCheckingSocket = (0, _asyncToGenerator2["default"])( /*#__PURE__*/_regenerator["default"].mark(function _callee3(successHandler, progressHandler, errorHandler) {
        var serverChoices, bestServer, bestServerTimeout, i, startTimestamp, ioInst, validStatus, _i;

        return _regenerator["default"].wrap(function _callee3$(_context3) {
          while (1) {
            switch (_context3.prev = _context3.next) {
              case 0:
                serverChoices = Rosemary.serverLists.availableSocketBackends;
                bestServer = "";
                bestServerTimeout = 12000;
                i = 0;

              case 4:
                if (!(i < serverChoices.length)) {
                  _context3.next = 23;
                  break;
                }

                if (i < 5) {
                  successHandler();
                }

                _context3.prev = 6;
                progressHandler("正在测试Socket(" + i + "/" + serverChoices.length + ") : " + serverChoices[i]);
                startTimestamp = Date.now();
                ioInst = _socket["default"].connect(serverChoices[i]);
                validStatus = false;
                ioInst.on("connect", function () {
                  validStatus = true;
                });
                _context3.next = 14;
                return new Promise(function (resolve) {
                  setTimeout(function () {
                    resolve();
                  }, 5000);
                });

              case 14:
                ioInst.disconnect();

                if (validStatus) {
                  bestServer = serverChoices[i];
                  bestServerTimeout = Date.now() - startTimestamp;
                }

                _context3.next = 20;
                break;

              case 18:
                _context3.prev = 18;
                _context3.t0 = _context3["catch"](6);

              case 20:
                i++;
                _context3.next = 4;
                break;

              case 23:
                if (serverChoices.length < 5) {
                  for (_i = 0; _i < 5 - serverChoices.length; _i++) {
                    successHandler();
                  }
                }

                if (!(bestServer === "")) {
                  _context3.next = 29;
                  break;
                }

                progressHandler("目前无可用的服务器，至" + serverChoices.length + "个服务器的连接均超时");
                throw new Error("目前无可用的服务器");

              case 29:
                localStorage.setItem("altina_socket_server", bestServer);

              case 30:
              case "end":
                return _context3.stop();
            }
          }
        }, _callee3, null, [[6, 18]]);
      }));

      function rosServerCheckingSocket(_x6, _x7, _x8) {
        return _rosServerCheckingSocket.apply(this, arguments);
      }

      return rosServerCheckingSocket;
    }()
  }, {
    key: "rosServerChecking",
    value: function () {
      var _rosServerChecking = (0, _asyncToGenerator2["default"])( /*#__PURE__*/_regenerator["default"].mark(function _callee4(successHandler, progressHandler, errorHandler) {
        var serverChoices, bestServer, bestServerTimeout, i, startTimestamp, _i2;

        return _regenerator["default"].wrap(function _callee4$(_context4) {
          while (1) {
            switch (_context4.prev = _context4.next) {
              case 0:
                console.log(Rosemary.serverLists);
                serverChoices = Rosemary.serverLists.availableBackends;
                bestServer = "";
                bestServerTimeout = 12000;
                i = 0;

              case 5:
                if (!(i < serverChoices.length)) {
                  _context4.next = 21;
                  break;
                }

                if (i < 5) {
                  successHandler();
                }

                _context4.prev = 7;
                progressHandler("正在测试连接(" + i + "/" + serverChoices.length + ") : " + serverChoices[i]);
                startTimestamp = Date.now();
                _context4.next = 12;
                return (0, _axios["default"])({
                  url: serverChoices[i] + "/handShake",
                  method: 'GET',
                  timeout: 5000
                });

              case 12:
                bestServer = serverChoices[i];
                bestServerTimeout = Date.now() - startTimestamp;
                _context4.next = 18;
                break;

              case 16:
                _context4.prev = 16;
                _context4.t0 = _context4["catch"](7);

              case 18:
                i++;
                _context4.next = 5;
                break;

              case 21:
                if (serverChoices.length < 5) {
                  for (_i2 = 0; _i2 < 5 - serverChoices.length; _i2++) {
                    successHandler();
                  }
                }

                if (!(bestServer === "")) {
                  _context4.next = 27;
                  break;
                }

                progressHandler("目前无可用的服务器，至" + serverChoices.length + "个服务器的连接均超时");
                throw new _sass.Exception("目前无可用的服务器");

              case 27:
                localStorage.setItem("altina_rapi_server", bestServer);

              case 28:
              case "end":
                return _context4.stop();
            }
          }
        }, _callee4, null, [[7, 16]]);
      }));

      function rosServerChecking(_x9, _x10, _x11) {
        return _rosServerChecking.apply(this, arguments);
      }

      return rosServerChecking;
    }()
  }, {
    key: "rosFontLoading",
    value: function () {
      var _rosFontLoading = (0, _asyncToGenerator2["default"])( /*#__PURE__*/_regenerator["default"].mark(function _callee5(successHandler, progressHandler, errorHandler) {
        var fontList, i, fontFace;
        return _regenerator["default"].wrap(function _callee5$(_context5) {
          while (1) {
            switch (_context5.prev = _context5.next) {
              case 0:
                fontList = [["Novecento", '/fonts/Geometos.ttf'], ['Geometos', '/fonts/Geometos.ttf'], ['SourceHan', '/fonts/SourceHanSansCN-Normal.otf'], ['SourceHanSerif', '/fonts/SourceHanSerifSC-Regular.otf'], ['SourceHanSerifHeavy', '/fonts/SourceHanSerifSC-Heavy.otf'], ['Bender', '/fonts/Bender.932867e7.ttf']];
                i = 0;

              case 2:
                if (!(i < fontList.length)) {
                  _context5.next = 12;
                  break;
                }

                progressHandler("正在加载字体 - " + fontList[i][0]);
                fontFace = new FontFace(fontList[i][0], "url('" + fontList[i][1] + "')");
                _context5.next = 7;
                return fontFace.load();

              case 7:
                document.fonts.add(fontFace);
                successHandler();

              case 9:
                i++;
                _context5.next = 2;
                break;

              case 12:
              case "end":
                return _context5.stop();
            }
          }
        }, _callee5);
      }));

      function rosFontLoading(_x12, _x13, _x14) {
        return _rosFontLoading.apply(this, arguments);
      }

      return rosFontLoading;
    }()
  }, {
    key: "rosFontLoaderMonitor",
    value: function rosFontLoaderMonitor(successHandler, progressHandler, errorHandler) {
      return new Promise(function (resolve, reject) {
        console.log(document.fonts);
        document.fonts.ready.then(resolve())["catch"](reject());
      });
    }
  }, {
    key: "rosServerListing",
    value: function () {
      var _rosServerListing = (0, _asyncToGenerator2["default"])( /*#__PURE__*/_regenerator["default"].mark(function _callee6(successHandler, progressHandler, errorHandler) {
        var _this = this;

        return _regenerator["default"].wrap(function _callee6$(_context6) {
          while (1) {
            switch (_context6.prev = _context6.next) {
              case 0:
                _context6.next = 2;
                return new Promise(function (resolve) {
                  setTimeout(function () {
                    resolve();
                  }, 1000);
                });

              case 2:
                _context6.next = 4;
                return (0, _axios["default"])({
                  url: "https://aeroraven.github.io/altina-backend/server_list.json",
                  method: 'GET'
                }).then(function (response) {
                  console.log(response);
                  console.log(_this);
                  Rosemary.serverLists = response.data;
                })["catch"](function (error) {
                  errorHandler("无法连接到aeroraven.github.io，请检查您的代理配置");
                });

              case 4:
              case "end":
                return _context6.stop();
            }
          }
        }, _callee6);
      }));

      function rosServerListing(_x15, _x16, _x17) {
        return _rosServerListing.apply(this, arguments);
      }

      return rosServerListing;
    }()
  }, {
    key: "rosExplorerCheck",
    value: function () {
      var _rosExplorerCheck = (0, _asyncToGenerator2["default"])( /*#__PURE__*/_regenerator["default"].mark(function _callee7() {
        return _regenerator["default"].wrap(function _callee7$(_context7) {
          while (1) {
            switch (_context7.prev = _context7.next) {
              case 0:
              case "end":
                return _context7.stop();
            }
          }
        }, _callee7);
      }));

      function rosExplorerCheck() {
        return _rosExplorerCheck.apply(this, arguments);
      }

      return rosExplorerCheck;
    }()
  }]);
  return Rosemary;
}();

exports["default"] = Rosemary;