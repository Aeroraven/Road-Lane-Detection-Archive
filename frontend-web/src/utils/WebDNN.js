!function(e, t) {
    "object" == typeof exports && "object" == typeof module ? module.exports = t() : "function" == typeof define && define.amd ? define([], t) : "object" == typeof exports ? exports.WebDNN = t() : e.WebDNN = t()
}(window, function() {
    var _StringfromCharCode = String.fromCharCode;
    return function(e) {
        function t(n) {
            if (r[n])
                return r[n].exports;
            var i = r[n] = {
                i: n,
                l: !1,
                exports: {}
            };
            return e[n].call(i.exports, i, i.exports, t),
            i.l = !0,
            i.exports
        }
        var r = {};
        return t.m = e,
        t.c = r,
        t.d = function(e, r, n) {
            t.o(e, r) || Object.defineProperty(e, r, {
                enumerable: !0,
                get: n
            })
        }
        ,
        t.r = function(e) {
            "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(e, Symbol.toStringTag, {
                value: "Module"
            }),
            Object.defineProperty(e, "__esModule", {
                value: !0
            })
        }
        ,
        t.t = function(e, r) {
            if (1 & r && (e = t(e)),
            8 & r)
                return e;
            if (4 & r && "object" == typeof e && e && e.__esModule)
                return e;
            var n = Object.create(null);
            if (t.r(n),
            Object.defineProperty(n, "default", {
                enumerable: !0,
                value: e
            }),
            2 & r && "string" != typeof e)
                for (var i in e)
                    t.d(n, i, function(t) {
                        return e[t]
                    }
                    .bind(null, i));
            return n
        }
        ,
        t.n = function(e) {
            var r = e && e.__esModule ? function() {
                return e.default
            }
            : function() {
                return e
            }
            ;
            return t.d(r, "a", r),
            r
        }
        ,
        t.o = function(e, t) {
            return Object.prototype.hasOwnProperty.call(e, t)
        }
        ,
        t.p = "",
        t(t.s = 19)
    }([function(e, t, r) {
        (function(t) {
            var r;
            /*!
    localForage -- Offline Storage, Improved
    Version 1.7.3
    https://localforage.github.io/localForage
    (c) 2013-2017 Mozilla, Apache License 2.0
*/
            /*!
    localForage -- Offline Storage, Improved
    Version 1.7.3
    https://localforage.github.io/localForage
    (c) 2013-2017 Mozilla, Apache License 2.0
*/
            !function(t) {
                e.exports = t()
            }(function() {
                return function e(t, n, i) {
                    function a(s, l) {
                        if (!n[s]) {
                            if (!t[s]) {
                                if (!l && ("function" == typeof r && r))
                                    return r(s, !0);
                                if (o)
                                    return o(s, !0);
                                var c = new Error("Cannot find module '" + s + "'");
                                throw c.code = "MODULE_NOT_FOUND",
                                c
                            }
                            var u = n[s] = {
                                exports: {}
                            };
                            t[s][0].call(u.exports, function(e) {
                                var r = t[s][1][e];
                                return a(r || e)
                            }, u, u.exports, e, t, n, i)
                        }
                        return n[s].exports
                    }
                    for (var o = "function" == typeof r && r, s = 0; s < i.length; s++)
                        a(i[s]);
                    return a
                }({
                    1: [function(e, r) {
                        (function(e) {
                            "use strict";
                            function t() {
                                c = !0;
                                for (var e, t, r = u.length; r; ) {
                                    for (t = u,
                                    u = [],
                                    e = -1; ++e < r; )
                                        t[e]();
                                    r = u.length
                                }
                                c = !1
                            }
                            var n, i = e.MutationObserver || e.WebKitMutationObserver;
                            if (i) {
                                var a = 0
                                  , o = new i(t)
                                  , s = e.document.createTextNode("");
                                o.observe(s, {
                                    characterData: !0
                                }),
                                n = function() {
                                    s.data = a = ++a % 2
                                }
                            } else if (e.setImmediate || void 0 === e.MessageChannel)
                                n = "document"in e && "onreadystatechange"in e.document.createElement("script") ? function() {
                                    var r = e.document.createElement("script");
                                    r.onreadystatechange = function() {
                                        t(),
                                        r.onreadystatechange = null,
                                        r.parentNode.removeChild(r),
                                        r = null
                                    }
                                    ,
                                    e.document.documentElement.appendChild(r)
                                }
                                : function() {
                                    setTimeout(t, 0)
                                }
                                ;
                            else {
                                var l = new e.MessageChannel;
                                l.port1.onmessage = t,
                                n = function() {
                                    l.port2.postMessage(0)
                                }
                            }
                            var c, u = [];
                            r.exports = function(e) {
                                1 !== u.push(e) || c || n()
                            }
                        }
                        ).call(this, void 0 === t ? "undefined" == typeof self ? "undefined" == typeof window ? {} : window : self : t)
                    }
                    , {}],
                    2: [function(e, t) {
                        "use strict";
                        function r() {}
                        function n(e) {
                            if ("function" != typeof e)
                                throw new TypeError("resolver must be a function");
                            this.state = d,
                            this.queue = [],
                            this.outcome = void 0,
                            e !== r && s(this, e)
                        }
                        function i(e, t, r) {
                            this.promise = e,
                            "function" == typeof t && (this.onFulfilled = t,
                            this.callFulfilled = this.otherCallFulfilled),
                            "function" == typeof r && (this.onRejected = r,
                            this.callRejected = this.otherCallRejected)
                        }
                        function a(e, t, r) {
                            c(function() {
                                var n;
                                try {
                                    n = t(r)
                                } catch (t) {
                                    return u.reject(e, t)
                                }
                                n === e ? u.reject(e, new TypeError("Cannot resolve promise with itself")) : u.resolve(e, n)
                            })
                        }
                        function o(e) {
                            var t = e && e.then;
                            if (e && ("object" == typeof e || "function" == typeof e) && "function" == typeof t)
                                return function() {
                                    t.apply(e, arguments)
                                }
                        }
                        function s(e, t) {
                            function r(t) {
                                i || (i = !0,
                                u.reject(e, t))
                            }
                            function n(t) {
                                i || (i = !0,
                                u.resolve(e, t))
                            }
                            var i = !1
                              , a = l(function() {
                                t(n, r)
                            });
                            "error" === a.status && r(a.value)
                        }
                        function l(e, t) {
                            var r = {};
                            try {
                                r.value = e(t),
                                r.status = "success"
                            } catch (t) {
                                r.status = "error",
                                r.value = t
                            }
                            return r
                        }
                        var c = e(1)
                          , u = {}
                          , h = ["REJECTED"]
                          , f = ["FULFILLED"]
                          , d = ["PENDING"];
                        t.exports = n,
                        n.prototype.catch = function(e) {
                            return this.then(null, e)
                        }
                        ,
                        n.prototype.then = function(e, t) {
                            if ("function" != typeof e && this.state === f || "function" != typeof t && this.state === h)
                                return this;
                            var n = new this.constructor(r);
                            this.state !== d ? a(n, this.state === f ? e : t, this.outcome) : this.queue.push(new i(n,e,t));
                            return n
                        }
                        ,
                        i.prototype.callFulfilled = function(e) {
                            u.resolve(this.promise, e)
                        }
                        ,
                        i.prototype.otherCallFulfilled = function(e) {
                            a(this.promise, this.onFulfilled, e)
                        }
                        ,
                        i.prototype.callRejected = function(e) {
                            u.reject(this.promise, e)
                        }
                        ,
                        i.prototype.otherCallRejected = function(e) {
                            a(this.promise, this.onRejected, e)
                        }
                        ,
                        u.resolve = function(e, t) {
                            var r = l(o, t);
                            if ("error" === r.status)
                                return u.reject(e, r.value);
                            var n = r.value;
                            if (n)
                                s(e, n);
                            else {
                                e.state = f,
                                e.outcome = t;
                                for (var i = -1, a = e.queue.length; ++i < a; )
                                    e.queue[i].callFulfilled(t)
                            }
                            return e
                        }
                        ,
                        u.reject = function(e, t) {
                            e.state = h,
                            e.outcome = t;
                            for (var r = -1, n = e.queue.length; ++r < n; )
                                e.queue[r].callRejected(t);
                            return e
                        }
                        ,
                        n.resolve = function(e) {
                            return e instanceof this ? e : u.resolve(new this(r), e)
                        }
                        ,
                        n.reject = function(e) {
                            var t = new this(r);
                            return u.reject(t, e)
                        }
                        ,
                        n.all = function(e) {
                            function t(e, t) {
                                n.resolve(e).then(function(e) {
                                    o[t] = e,
                                    ++s !== i || a || (a = !0,
                                    u.resolve(c, o))
                                }, function(e) {
                                    a || (a = !0,
                                    u.reject(c, e))
                                })
                            }
                            var n = this;
                            if ("[object Array]" !== Object.prototype.toString.call(e))
                                return this.reject(new TypeError("must be an array"));
                            var i = e.length
                              , a = !1;
                            if (!i)
                                return this.resolve([]);
                            for (var o = Array(i), s = 0, l = -1, c = new this(r); ++l < i; )
                                t(e[l], l);
                            return c
                        }
                        ,
                        n.race = function(e) {
                            function t(e) {
                                n.resolve(e).then(function(e) {
                                    a || (a = !0,
                                    u.resolve(s, e))
                                }, function(e) {
                                    a || (a = !0,
                                    u.reject(s, e))
                                })
                            }
                            var n = this;
                            if ("[object Array]" !== Object.prototype.toString.call(e))
                                return this.reject(new TypeError("must be an array"));
                            var i = e.length
                              , a = !1;
                            if (!i)
                                return this.resolve([]);
                            for (var o = -1, s = new this(r); ++o < i; )
                                t(e[o]);
                            return s
                        }
                    }
                    , {
                        1: 1
                    }],
                    3: [function(e) {
                        (function(t) {
                            "use strict";
                            "function" != typeof t.Promise && (t.Promise = e(2))
                        }
                        ).call(this, void 0 === t ? "undefined" == typeof self ? "undefined" == typeof window ? {} : window : self : t)
                    }
                    , {
                        2: 2
                    }],
                    4: [function(e, t) {
                        "use strict";
                        function r(e, t) {
                            e = e || [],
                            t = t || {};
                            try {
                                return new Blob(e,t)
                            } catch (i) {
                                if ("TypeError" !== i.name)
                                    throw i;
                                for (var r = new ("undefined" == typeof BlobBuilder ? "undefined" == typeof MSBlobBuilder ? "undefined" == typeof MozBlobBuilder ? WebKitBlobBuilder : MozBlobBuilder : MSBlobBuilder : BlobBuilder), n = 0; n < e.length; n += 1)
                                    r.append(e[n]);
                                return r.getBlob(t.type)
                            }
                        }
                        function n(e, t) {
                            t && e.then(function(e) {
                                t(null, e)
                            }, function(e) {
                                t(e)
                            })
                        }
                        function i(e, t, r) {
                            "function" == typeof t && e.then(t),
                            "function" == typeof r && e.catch(r)
                        }
                        function a(e) {
                            return "string" != typeof e && (console.warn(e + " used as a key, but it is not a string."),
                            e += ""),
                            e
                        }
                        function o() {
                            if (arguments.length && "function" == typeof arguments[arguments.length - 1])
                                return arguments[arguments.length - 1]
                        }
                        function s(e) {
                            return "boolean" == typeof R ? C.resolve(R) : function(e) {
                                return new C(function(t) {
                                    var n = e.transaction(B, P)
                                      , i = r([""]);
                                    n.objectStore(B).put(i, "key"),
                                    n.onabort = function(e) {
                                        e.preventDefault(),
                                        e.stopPropagation(),
                                        t(!1)
                                    }
                                    ,
                                    n.oncomplete = function() {
                                        var e = navigator.userAgent.match(/Chrome\/(\d+)/)
                                          , r = navigator.userAgent.match(/Edge\//);
                                        t(r || !e || 43 <= parseInt(e[1], 10))
                                    }
                                }
                                ).catch(function() {
                                    return !1
                                })
                            }(e).then(function(e) {
                                return R = e
                            })
                        }
                        function l(e) {
                            var t = D[e.name]
                              , r = {};
                            r.promise = new C(function(e, t) {
                                r.resolve = e,
                                r.reject = t
                            }
                            ),
                            t.deferredOperations.push(r),
                            t.dbReady = t.dbReady ? t.dbReady.then(function() {
                                return r.promise
                            }) : r.promise
                        }
                        function c(e) {
                            var t = D[e.name].deferredOperations.pop();
                            if (t)
                                return t.resolve(),
                                t.promise
                        }
                        function u(e, t) {
                            var r = D[e.name].deferredOperations.pop();
                            if (r)
                                return r.reject(t),
                                r.promise
                        }
                        function h(e, t) {
                            return new C(function(r, n) {
                                if (D[e.name] = D[e.name] || {
                                    forages: [],
                                    db: null,
                                    dbReady: null,
                                    deferredOperations: []
                                },
                                e.db) {
                                    if (!t)
                                        return r(e.db);
                                    l(e),
                                    e.db.close()
                                }
                                var i = [e.name];
                                t && i.push(e.version);
                                var a = I.open.apply(I, i);
                                t && (a.onupgradeneeded = function(t) {
                                    var r = a.result;
                                    try {
                                        r.createObjectStore(e.storeName),
                                        1 >= t.oldVersion && r.createObjectStore(B)
                                    } catch (r) {
                                        if ("ConstraintError" !== r.name)
                                            throw r;
                                        console.warn('The database "' + e.name + '" has been upgraded from version ' + t.oldVersion + " to version " + t.newVersion + ', but the storage "' + e.storeName + '" already exists.')
                                    }
                                }
                                ),
                                a.onerror = function(e) {
                                    e.preventDefault(),
                                    n(a.error)
                                }
                                ,
                                a.onsuccess = function() {
                                    r(a.result),
                                    c(e)
                                }
                            }
                            )
                        }
                        function f(e) {
                            return h(e, !1)
                        }
                        function d(e) {
                            return h(e, !0)
                        }
                        function p(e, t) {
                            if (!e.db)
                                return !0;
                            var r = !e.db.objectStoreNames.contains(e.storeName)
                              , n = e.version < e.db.version
                              , i = e.version > e.db.version;
                            if (n && (e.version !== t && console.warn('The database "' + e.name + "\" can't be downgraded from version " + e.db.version + " to version " + e.version + "."),
                            e.version = e.db.version),
                            i || r) {
                                if (r) {
                                    var a = e.db.version + 1;
                                    a > e.version && (e.version = a)
                                }
                                return !0
                            }
                            return !1
                        }
                        function _(e) {
                            var t = function(e) {
                                for (var t = e.length, r = new ArrayBuffer(t), n = new Uint8Array(r), i = 0; i < t; i++)
                                    n[i] = e.charCodeAt(i);
                                return r
                            }(atob(e.data));
                            return r([t], {
                                type: e.type
                            })
                        }
                        function m(e) {
                            return e && e.__local_forage_encoded_blob
                        }
                        function w(e) {
                            var t = this
                              , r = t._initReady().then(function() {
                                var e = D[t._dbInfo.name];
                                if (e && e.dbReady)
                                    return e.dbReady
                            });
                            return i(r, e, e),
                            r
                        }
                        function b(e, t, r, n) {
                            void 0 === n && (n = 1);
                            try {
                                var i = e.db.transaction(e.storeName, t);
                                r(null, i)
                            } catch (i) {
                                if (0 < n && (!e.db || "InvalidStateError" === i.name || "NotFoundError" === i.name))
                                    return C.resolve().then(function() {
                                        if (!e.db || "NotFoundError" === i.name && !e.db.objectStoreNames.contains(e.storeName) && e.version <= e.db.version)
                                            return e.db && (e.version = e.db.version + 1),
                                            d(e)
                                    }).then(function() {
                                        return function(e) {
                                            l(e);
                                            for (var t, r = D[e.name], n = r.forages, i = 0; i < n.length; i++)
                                                (t = n[i])._dbInfo.db && (t._dbInfo.db.close(),
                                                t._dbInfo.db = null);
                                            return e.db = null,
                                            f(e).then(function(t) {
                                                return e.db = t,
                                                p(e) ? d(e) : t
                                            }).then(function(t) {
                                                e.db = r.db = t;
                                                for (var i = 0; i < n.length; i++)
                                                    n[i]._dbInfo.db = t
                                            }).catch(function(t) {
                                                throw u(e, t),
                                                t
                                            })
                                        }(e).then(function() {
                                            b(e, t, r, n - 1)
                                        })
                                    }).catch(r);
                                r(i)
                            }
                        }
                        function g(e) {
                            var t, r, n, i, a, o = .75 * e.length, s = e.length, l = 0;
                            "=" === e[e.length - 1] && (o--,
                            "=" === e[e.length - 2] && o--);
                            var c = new ArrayBuffer(o)
                              , u = new Uint8Array(c);
                            for (t = 0; t < s; t += 4)
                                r = F.indexOf(e[t]),
                                n = F.indexOf(e[t + 1]),
                                i = F.indexOf(e[t + 2]),
                                a = F.indexOf(e[t + 3]),
                                u[l++] = r << 2 | n >> 4,
                                u[l++] = (15 & n) << 4 | i >> 2,
                                u[l++] = (3 & i) << 6 | 63 & a;
                            return c
                        }
                        function y(e) {
                            var t, r = new Uint8Array(e), n = "";
                            for (t = 0; t < r.length; t += 3)
                                n += F[r[t] >> 2],
                                n += F[(3 & r[t]) << 4 | r[t + 1] >> 4],
                                n += F[(15 & r[t + 1]) << 2 | r[t + 2] >> 6],
                                n += F[63 & r[t + 2]];
                            return 2 == r.length % 3 ? n = n.substring(0, n.length - 1) + "=" : 1 == r.length % 3 && (n = n.substring(0, n.length - 2) + "=="),
                            n
                        }
                        function v(e, t, r, n) {
                            e.executeSql("CREATE TABLE IF NOT EXISTS " + t.storeName + " (id INTEGER PRIMARY KEY, key unique, value)", [], r, n)
                        }
                        function E(e, t, r, n, i, a) {
                            e.executeSql(r, n, i, function(e, o) {
                                o.code === o.SYNTAX_ERR ? e.executeSql("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", [t.storeName], function(e, s) {
                                    s.rows.length ? a(e, o) : v(e, t, function() {
                                        e.executeSql(r, n, i, a)
                                    }, a)
                                }, a) : a(e, o)
                            }, a)
                        }
                        function k(e, t) {
                            var r = e.name + "/";
                            return e.storeName !== t.storeName && (r += e.storeName + "/"),
                            r
                        }
                        function x() {
                            return !function() {
                                var e = "_localforage_support_test";
                                try {
                                    return localStorage.setItem(e, !0),
                                    localStorage.removeItem(e),
                                    !1
                                } catch (e) {
                                    return !0
                                }
                            }() || 0 < localStorage.length
                        }
                        function A(e, t) {
                            e[t] = function() {
                                var r = arguments;
                                return e.ready().then(function() {
                                    return e[t].apply(e, r)
                                })
                            }
                        }
                        function S() {
                            for (var e, t = 1; t < arguments.length; t++)
                                if (e = arguments[t])
                                    for (var r in e)
                                        e.hasOwnProperty(r) && (arguments[0][r] = q(e[r]) ? e[r].slice() : e[r]);
                            return arguments[0]
                        }
                        var T = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(e) {
                            return typeof e
                        }
                        : function(e) {
                            return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
                        }
                          , I = function() {
                            try {
                                if ("undefined" != typeof indexedDB)
                                    return indexedDB;
                                if ("undefined" != typeof webkitIndexedDB)
                                    return webkitIndexedDB;
                                if ("undefined" != typeof mozIndexedDB)
                                    return mozIndexedDB;
                                if ("undefined" != typeof OIndexedDB)
                                    return OIndexedDB;
                                if ("undefined" != typeof msIndexedDB)
                                    return msIndexedDB
                            } catch (e) {}
                        }();
                        "undefined" == typeof Promise && e(3);
                        var C = Promise
                          , B = "local-forage-detect-blob-support"
                          , R = void 0
                          , D = {}
                          , z = Object.prototype.toString
                          , N = "readonly"
                          , P = "readwrite"
                          , O = {
                            _driver: "asyncStorage",
                            _initStorage: function(e) {
                                function t() {
                                    return C.resolve()
                                }
                                var r = this
                                  , n = {
                                    db: null
                                };
                                if (e)
                                    for (var i in e)
                                        n[i] = e[i];
                                var a = D[n.name];
                                a || (a = {
                                    forages: [],
                                    db: null,
                                    dbReady: null,
                                    deferredOperations: []
                                },
                                D[n.name] = a),
                                a.forages.push(r),
                                r._initReady || (r._initReady = r.ready,
                                r.ready = w);
                                for (var o, s = [], l = 0; l < a.forages.length; l++)
                                    (o = a.forages[l]) !== r && s.push(o._initReady().catch(t));
                                var c = a.forages.slice(0);
                                return C.all(s).then(function() {
                                    return n.db = a.db,
                                    f(n)
                                }).then(function(e) {
                                    return n.db = e,
                                    p(n, r._defaultConfig.version) ? d(n) : e
                                }).then(function(e) {
                                    n.db = a.db = e,
                                    r._dbInfo = n;
                                    for (var t, i = 0; i < c.length; i++)
                                        (t = c[i]) !== r && (t._dbInfo.db = n.db,
                                        t._dbInfo.version = n.version)
                                })
                            },
                            _support: function() {
                                try {
                                    if (!I)
                                        return !1;
                                    var e = "undefined" != typeof openDatabase && /(Safari|iPhone|iPad|iPod)/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent) && !/BlackBerry/.test(navigator.platform)
                                      , t = "function" == typeof fetch && -1 !== fetch.toString().indexOf("[native code");
                                    return (!e || t) && "undefined" != typeof indexedDB && "undefined" != typeof IDBKeyRange
                                } catch (t) {
                                    return !1
                                }
                            }(),
                            iterate: function(e, t) {
                                var r = this
                                  , i = new C(function(t, n) {
                                    r.ready().then(function() {
                                        b(r._dbInfo, N, function(i, a) {
                                            if (i)
                                                return n(i);
                                            try {
                                                var o = a.objectStore(r._dbInfo.storeName).openCursor()
                                                  , s = 1;
                                                o.onsuccess = function() {
                                                    var r = o.result;
                                                    if (r) {
                                                        var n = r.value;
                                                        m(n) && (n = _(n));
                                                        var i = e(n, r.key, s++);
                                                        void 0 === i ? r.continue() : t(i)
                                                    } else
                                                        t()
                                                }
                                                ,
                                                o.onerror = function() {
                                                    n(o.error)
                                                }
                                            } catch (e) {
                                                n(e)
                                            }
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            getItem: function(e, t) {
                                var r = this;
                                e = a(e);
                                var i = new C(function(t, n) {
                                    r.ready().then(function() {
                                        b(r._dbInfo, N, function(i, a) {
                                            if (i)
                                                return n(i);
                                            try {
                                                var o = a.objectStore(r._dbInfo.storeName).get(e);
                                                o.onsuccess = function() {
                                                    var e = o.result;
                                                    void 0 === e && (e = null),
                                                    m(e) && (e = _(e)),
                                                    t(e)
                                                }
                                                ,
                                                o.onerror = function() {
                                                    n(o.error)
                                                }
                                            } catch (e) {
                                                n(e)
                                            }
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            setItem: function(e, t, r) {
                                var i = this;
                                e = a(e);
                                var o = new C(function(r, n) {
                                    var a;
                                    i.ready().then(function() {
                                        return a = i._dbInfo,
                                        "[object Blob]" === z.call(t) ? s(a.db).then(function(e) {
                                            return e ? t : function(e) {
                                                return new C(function(t, r) {
                                                    var n = new FileReader;
                                                    n.onerror = r,
                                                    n.onloadend = function(r) {
                                                        var n = btoa(r.target.result || "");
                                                        t({
                                                            __local_forage_encoded_blob: !0,
                                                            data: n,
                                                            type: e.type
                                                        })
                                                    }
                                                    ,
                                                    n.readAsBinaryString(e)
                                                }
                                                )
                                            }(t)
                                        }) : t
                                    }).then(function(t) {
                                        b(i._dbInfo, P, function(a, o) {
                                            if (a)
                                                return n(a);
                                            try {
                                                var s = o.objectStore(i._dbInfo.storeName);
                                                null === t && (t = void 0);
                                                var l = s.put(t, e);
                                                o.oncomplete = function() {
                                                    void 0 === t && (t = null),
                                                    r(t)
                                                }
                                                ,
                                                o.onabort = o.onerror = function() {
                                                    var e = l.error ? l.error : l.transaction.error;
                                                    n(e)
                                                }
                                            } catch (e) {
                                                n(e)
                                            }
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(o, r),
                                o
                            },
                            removeItem: function(e, t) {
                                var r = this;
                                e = a(e);
                                var i = new C(function(t, n) {
                                    r.ready().then(function() {
                                        b(r._dbInfo, P, function(i, a) {
                                            if (i)
                                                return n(i);
                                            try {
                                                var o = a.objectStore(r._dbInfo.storeName).delete(e);
                                                a.oncomplete = function() {
                                                    t()
                                                }
                                                ,
                                                a.onerror = function() {
                                                    n(o.error)
                                                }
                                                ,
                                                a.onabort = function() {
                                                    var e = o.error ? o.error : o.transaction.error;
                                                    n(e)
                                                }
                                            } catch (e) {
                                                n(e)
                                            }
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            clear: function(e) {
                                var t = this
                                  , r = new C(function(e, r) {
                                    t.ready().then(function() {
                                        b(t._dbInfo, P, function(n, i) {
                                            if (n)
                                                return r(n);
                                            try {
                                                var a = i.objectStore(t._dbInfo.storeName).clear();
                                                i.oncomplete = function() {
                                                    e()
                                                }
                                                ,
                                                i.onabort = i.onerror = function() {
                                                    var e = a.error ? a.error : a.transaction.error;
                                                    r(e)
                                                }
                                            } catch (e) {
                                                r(e)
                                            }
                                        })
                                    }).catch(r)
                                }
                                );
                                return n(r, e),
                                r
                            },
                            length: function(e) {
                                var t = this
                                  , r = new C(function(e, r) {
                                    t.ready().then(function() {
                                        b(t._dbInfo, N, function(n, i) {
                                            if (n)
                                                return r(n);
                                            try {
                                                var a = i.objectStore(t._dbInfo.storeName).count();
                                                a.onsuccess = function() {
                                                    e(a.result)
                                                }
                                                ,
                                                a.onerror = function() {
                                                    r(a.error)
                                                }
                                            } catch (e) {
                                                r(e)
                                            }
                                        })
                                    }).catch(r)
                                }
                                );
                                return n(r, e),
                                r
                            },
                            key: function(e, t) {
                                var r = this
                                  , i = new C(function(t, n) {
                                    return 0 > e ? void t(null) : void r.ready().then(function() {
                                        b(r._dbInfo, N, function(i, a) {
                                            if (i)
                                                return n(i);
                                            try {
                                                var o = a.objectStore(r._dbInfo.storeName)
                                                  , s = !1
                                                  , l = o.openCursor();
                                                l.onsuccess = function() {
                                                    var r = l.result;
                                                    return r ? void (0 === e ? t(r.key) : s ? t(r.key) : (s = !0,
                                                    r.advance(e))) : void t(null)
                                                }
                                                ,
                                                l.onerror = function() {
                                                    n(l.error)
                                                }
                                            } catch (e) {
                                                n(e)
                                            }
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            keys: function(e) {
                                var t = this
                                  , r = new C(function(e, r) {
                                    t.ready().then(function() {
                                        b(t._dbInfo, N, function(n, i) {
                                            if (n)
                                                return r(n);
                                            try {
                                                var a = i.objectStore(t._dbInfo.storeName).openCursor()
                                                  , o = [];
                                                a.onsuccess = function() {
                                                    var t = a.result;
                                                    return t ? (o.push(t.key),
                                                    void t.continue()) : void e(o)
                                                }
                                                ,
                                                a.onerror = function() {
                                                    r(a.error)
                                                }
                                            } catch (e) {
                                                r(e)
                                            }
                                        })
                                    }).catch(r)
                                }
                                );
                                return n(r, e),
                                r
                            },
                            dropInstance: function(e, t) {
                                t = o.apply(this, arguments);
                                var r = this.config();
                                (e = "function" != typeof e && e || {}).name || (e.name = e.name || r.name,
                                e.storeName = e.storeName || r.storeName);
                                var i;
                                if (e.name) {
                                    var a = e.name === r.name && this._dbInfo.db ? C.resolve(this._dbInfo.db) : f(e).then(function(t) {
                                        var r = D[e.name]
                                          , n = r.forages;
                                        r.db = t;
                                        for (var i = 0; i < n.length; i++)
                                            n[i]._dbInfo.db = t;
                                        return t
                                    });
                                    i = e.storeName ? a.then(function(t) {
                                        if (t.objectStoreNames.contains(e.storeName)) {
                                            var r = t.version + 1;
                                            l(e);
                                            var n = D[e.name]
                                              , i = n.forages;
                                            t.close();
                                            for (var a, o = 0; o < i.length; o++)
                                                (a = i[o])._dbInfo.db = null,
                                                a._dbInfo.version = r;
                                            return new C(function(t, n) {
                                                var i = I.open(e.name, r);
                                                i.onerror = function(e) {
                                                    i.result.close(),
                                                    n(e)
                                                }
                                                ,
                                                i.onupgradeneeded = function() {
                                                    i.result.deleteObjectStore(e.storeName)
                                                }
                                                ,
                                                i.onsuccess = function() {
                                                    var e = i.result;
                                                    e.close(),
                                                    t(e)
                                                }
                                            }
                                            ).then(function(e) {
                                                n.db = e;
                                                for (var t, r = 0; r < i.length; r++)
                                                    (t = i[r])._dbInfo.db = e,
                                                    c(t._dbInfo)
                                            }).catch(function(t) {
                                                throw (u(e, t) || C.resolve()).catch(function() {}),
                                                t
                                            })
                                        }
                                    }) : a.then(function(t) {
                                        l(e);
                                        var r = D[e.name]
                                          , n = r.forages;
                                        t.close();
                                        for (var i = 0; i < n.length; i++)
                                            n[i]._dbInfo.db = null;
                                        return new C(function(t, r) {
                                            var n = I.deleteDatabase(e.name);
                                            n.onerror = n.onblocked = function(e) {
                                                var t = n.result;
                                                t && t.close(),
                                                r(e)
                                            }
                                            ,
                                            n.onsuccess = function() {
                                                var e = n.result;
                                                e && e.close(),
                                                t(e)
                                            }
                                        }
                                        ).then(function(e) {
                                            r.db = e;
                                            for (var t = 0; t < n.length; t++)
                                                c(n[t]._dbInfo)
                                        }).catch(function(t) {
                                            throw (u(e, t) || C.resolve()).catch(function() {}),
                                            t
                                        })
                                    })
                                } else
                                    i = C.reject("Invalid arguments");
                                return n(i, t),
                                i
                            }
                        }
                          , F = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
                          , L = /^~~local_forage_type~([^~]+)~/
                          , U = "__lfsc__:"
                          , M = U.length
                          , j = "arbf"
                          , W = "blob"
                          , H = "si08"
                          , V = M + j.length
                          , $ = Object.prototype.toString
                          , G = {
                            serialize: function(e, t) {
                                var r = "";
                                if (e && (r = $.call(e)),
                                e && ("[object ArrayBuffer]" === r || e.buffer && "[object ArrayBuffer]" === $.call(e.buffer))) {
                                    var n, i = U;
                                    e instanceof ArrayBuffer ? (n = e,
                                    i += j) : (n = e.buffer,
                                    "[object Int8Array]" === r ? i += H : "[object Uint8Array]" === r ? i += "ui08" : "[object Uint8ClampedArray]" === r ? i += "uic8" : "[object Int16Array]" === r ? i += "si16" : "[object Uint16Array]" === r ? i += "ur16" : "[object Int32Array]" === r ? i += "si32" : "[object Uint32Array]" === r ? i += "ui32" : "[object Float32Array]" === r ? i += "fl32" : "[object Float64Array]" === r ? i += "fl64" : t(new Error("Failed to get type for BinaryArray"))),
                                    t(i + y(n))
                                } else if ("[object Blob]" === r) {
                                    var a = new FileReader;
                                    a.onload = function() {
                                        var r = "~~local_forage_type~" + e.type + "~" + y(this.result);
                                        t(U + W + r)
                                    }
                                    ,
                                    a.readAsArrayBuffer(e)
                                } else
                                    try {
                                        t(JSON.stringify(e))
                                    } catch (i) {
                                        console.error("Couldn't convert value into a JSON string: ", e),
                                        t(null, i)
                                    }
                            },
                            deserialize: function(e) {
                                if (e.substring(0, M) !== U)
                                    return JSON.parse(e);
                                var t, n = e.substring(V), i = e.substring(M, V);
                                if (i === W && L.test(n)) {
                                    var a = n.match(L);
                                    t = a[1],
                                    n = n.substring(a[0].length)
                                }
                                var o = g(n);
                                switch (i) {
                                case j:
                                    return o;
                                case W:
                                    return r([o], {
                                        type: t
                                    });
                                case H:
                                    return new Int8Array(o);
                                case "ui08":
                                    return new Uint8Array(o);
                                case "uic8":
                                    return new Uint8ClampedArray(o);
                                case "si16":
                                    return new Int16Array(o);
                                case "ur16":
                                    return new Uint16Array(o);
                                case "si32":
                                    return new Int32Array(o);
                                case "ui32":
                                    return new Uint32Array(o);
                                case "fl32":
                                    return new Float32Array(o);
                                case "fl64":
                                    return new Float64Array(o);
                                default:
                                    throw new Error("Unkown type: " + i)
                                }
                            },
                            stringToBuffer: g,
                            bufferToString: y
                        }
                          , X = {
                            _driver: "webSQLStorage",
                            _initStorage: function(e) {
                                var t = this
                                  , r = {
                                    db: null
                                };
                                if (e)
                                    for (var n in e)
                                        r[n] = "string" == typeof e[n] ? e[n] : e[n].toString();
                                var i = new C(function(e, n) {
                                    try {
                                        r.db = openDatabase(r.name, r.version + "", r.description, r.size)
                                    } catch (e) {
                                        return n(e)
                                    }
                                    r.db.transaction(function(i) {
                                        v(i, r, function() {
                                            t._dbInfo = r,
                                            e()
                                        }, function(e, t) {
                                            n(t)
                                        })
                                    }, n)
                                }
                                );
                                return r.serializer = G,
                                i
                            },
                            _support: "function" == typeof openDatabase,
                            iterate: function(e, t) {
                                var r = this
                                  , i = new C(function(t, n) {
                                    r.ready().then(function() {
                                        var i = r._dbInfo;
                                        i.db.transaction(function(r) {
                                            E(r, i, "SELECT * FROM " + i.storeName, [], function(r, n) {
                                                for (var a = n.rows, o = a.length, s = 0; s < o; s++) {
                                                    var l = a.item(s)
                                                      , c = l.value;
                                                    if (c && (c = i.serializer.deserialize(c)),
                                                    void 0 !== (c = e(c, l.key, s + 1)))
                                                        return void t(c)
                                                }
                                                t()
                                            }, function(e, t) {
                                                n(t)
                                            })
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            getItem: function(e, t) {
                                var r = this;
                                e = a(e);
                                var i = new C(function(t, n) {
                                    r.ready().then(function() {
                                        var i = r._dbInfo;
                                        i.db.transaction(function(r) {
                                            E(r, i, "SELECT * FROM " + i.storeName + " WHERE key = ? LIMIT 1", [e], function(e, r) {
                                                var n = r.rows.length ? r.rows.item(0).value : null;
                                                n && (n = i.serializer.deserialize(n)),
                                                t(n)
                                            }, function(e, t) {
                                                n(t)
                                            })
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            setItem: function(e, t, r) {
                                return function e(t, r, i, o) {
                                    var s = this;
                                    t = a(t);
                                    var l = new C(function(n, a) {
                                        s.ready().then(function() {
                                            void 0 === r && (r = null);
                                            var l = r
                                              , c = s._dbInfo;
                                            c.serializer.serialize(r, function(r, u) {
                                                u ? a(u) : c.db.transaction(function(e) {
                                                    E(e, c, "INSERT OR REPLACE INTO " + c.storeName + " (key, value) VALUES (?, ?)", [t, r], function() {
                                                        n(l)
                                                    }, function(e, t) {
                                                        a(t)
                                                    })
                                                }, function(r) {
                                                    if (r.code === r.QUOTA_ERR) {
                                                        if (0 < o)
                                                            return void n(e.apply(s, [t, l, i, o - 1]));
                                                        a(r)
                                                    }
                                                })
                                            })
                                        }).catch(a)
                                    }
                                    );
                                    return n(l, i),
                                    l
                                }
                                .apply(this, [e, t, r, 1])
                            },
                            removeItem: function(e, t) {
                                var r = this;
                                e = a(e);
                                var i = new C(function(t, n) {
                                    r.ready().then(function() {
                                        var i = r._dbInfo;
                                        i.db.transaction(function(r) {
                                            E(r, i, "DELETE FROM " + i.storeName + " WHERE key = ?", [e], function() {
                                                t()
                                            }, function(e, t) {
                                                n(t)
                                            })
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            clear: function(e) {
                                var t = this
                                  , r = new C(function(e, r) {
                                    t.ready().then(function() {
                                        var n = t._dbInfo;
                                        n.db.transaction(function(t) {
                                            E(t, n, "DELETE FROM " + n.storeName, [], function() {
                                                e()
                                            }, function(e, t) {
                                                r(t)
                                            })
                                        })
                                    }).catch(r)
                                }
                                );
                                return n(r, e),
                                r
                            },
                            length: function(e) {
                                var t = this
                                  , r = new C(function(e, r) {
                                    t.ready().then(function() {
                                        var n = t._dbInfo;
                                        n.db.transaction(function(t) {
                                            E(t, n, "SELECT COUNT(key) as c FROM " + n.storeName, [], function(t, r) {
                                                var n = r.rows.item(0).c;
                                                e(n)
                                            }, function(e, t) {
                                                r(t)
                                            })
                                        })
                                    }).catch(r)
                                }
                                );
                                return n(r, e),
                                r
                            },
                            key: function(e, t) {
                                var r = this
                                  , i = new C(function(t, n) {
                                    r.ready().then(function() {
                                        var i = r._dbInfo;
                                        i.db.transaction(function(r) {
                                            E(r, i, "SELECT key FROM " + i.storeName + " WHERE id = ? LIMIT 1", [e + 1], function(e, r) {
                                                var n = r.rows.length ? r.rows.item(0).key : null;
                                                t(n)
                                            }, function(e, t) {
                                                n(t)
                                            })
                                        })
                                    }).catch(n)
                                }
                                );
                                return n(i, t),
                                i
                            },
                            keys: function(e) {
                                var t = this
                                  , r = new C(function(e, r) {
                                    t.ready().then(function() {
                                        var n = t._dbInfo;
                                        n.db.transaction(function(t) {
                                            E(t, n, "SELECT key FROM " + n.storeName, [], function(t, r) {
                                                for (var n = [], i = 0; i < r.rows.length; i++)
                                                    n.push(r.rows.item(i).key);
                                                e(n)
                                            }, function(e, t) {
                                                r(t)
                                            })
                                        })
                                    }).catch(r)
                                }
                                );
                                return n(r, e),
                                r
                            },
                            dropInstance: function(e, t) {
                                t = o.apply(this, arguments);
                                var r = this.config();
                                (e = "function" != typeof e && e || {}).name || (e.name = e.name || r.name,
                                e.storeName = e.storeName || r.storeName);
                                var i, a = this;
                                return i = e.name ? new C(function(t) {
                                    var n;
                                    n = e.name === r.name ? a._dbInfo.db : openDatabase(e.name, "", "", 0),
                                    e.storeName ? t({
                                        db: n,
                                        storeNames: [e.storeName]
                                    }) : t(function(e) {
                                        return new C(function(t, r) {
                                            e.transaction(function(n) {
                                                n.executeSql("SELECT name FROM sqlite_master WHERE type='table' AND name <> '__WebKitDatabaseInfoTable__'", [], function(r, n) {
                                                    for (var i = [], a = 0; a < n.rows.length; a++)
                                                        i.push(n.rows.item(a).name);
                                                    t({
                                                        db: e,
                                                        storeNames: i
                                                    })
                                                }, function(e, t) {
                                                    r(t)
                                                })
                                            }, function(e) {
                                                r(e)
                                            })
                                        }
                                        )
                                    }(n))
                                }
                                ).then(function(e) {
                                    return new C(function(t, r) {
                                        e.db.transaction(function(n) {
                                            function i(e) {
                                                return new C(function(t, r) {
                                                    n.executeSql("DROP TABLE IF EXISTS " + e, [], function() {
                                                        t()
                                                    }, function(e, t) {
                                                        r(t)
                                                    })
                                                }
                                                )
                                            }
                                            for (var a = [], o = 0, s = e.storeNames.length; o < s; o++)
                                                a.push(i(e.storeNames[o]));
                                            C.all(a).then(function() {
                                                t()
                                            }).catch(function(e) {
                                                r(e)
                                            })
                                        }, function(e) {
                                            r(e)
                                        })
                                    }
                                    )
                                }) : C.reject("Invalid arguments"),
                                n(i, t),
                                i
                            }
                        }
                          , Y = {
                            _driver: "localStorageWrapper",
                            _initStorage: function(e) {
                                var t = {};
                                if (e)
                                    for (var r in e)
                                        t[r] = e[r];
                                return t.keyPrefix = k(e, this._defaultConfig),
                                x() ? (this._dbInfo = t,
                                t.serializer = G,
                                C.resolve()) : C.reject()
                            },
                            _support: function() {
                                try {
                                    return "undefined" != typeof localStorage && "setItem"in localStorage && !!localStorage.setItem
                                } catch (e) {
                                    return !1
                                }
                            }(),
                            iterate: function(e, t) {
                                var r = this
                                  , i = r.ready().then(function() {
                                    for (var t, n = r._dbInfo, i = n.keyPrefix, a = i.length, o = localStorage.length, s = 1, l = 0; l < o; l++)
                                        if (0 === (t = localStorage.key(l)).indexOf(i)) {
                                            var c = localStorage.getItem(t);
                                            if (c && (c = n.serializer.deserialize(c)),
                                            void 0 !== (c = e(c, t.substring(a), s++)))
                                                return c
                                        }
                                });
                                return n(i, t),
                                i
                            },
                            getItem: function(e, t) {
                                var r = this;
                                e = a(e);
                                var i = r.ready().then(function() {
                                    var t = r._dbInfo
                                      , n = localStorage.getItem(t.keyPrefix + e);
                                    return n && (n = t.serializer.deserialize(n)),
                                    n
                                });
                                return n(i, t),
                                i
                            },
                            setItem: function(e, t, r) {
                                var i = this;
                                e = a(e);
                                var o = i.ready().then(function() {
                                    void 0 === t && (t = null);
                                    var r = t;
                                    return new C(function(n, a) {
                                        var o = i._dbInfo;
                                        o.serializer.serialize(t, function(t, i) {
                                            if (i)
                                                a(i);
                                            else
                                                try {
                                                    localStorage.setItem(o.keyPrefix + e, t),
                                                    n(r)
                                                } catch (t) {
                                                    ("QuotaExceededError" === t.name || "NS_ERROR_DOM_QUOTA_REACHED" === t.name) && a(t),
                                                    a(t)
                                                }
                                        })
                                    }
                                    )
                                });
                                return n(o, r),
                                o
                            },
                            removeItem: function(e, t) {
                                var r = this;
                                e = a(e);
                                var i = r.ready().then(function() {
                                    var t = r._dbInfo;
                                    localStorage.removeItem(t.keyPrefix + e)
                                });
                                return n(i, t),
                                i
                            },
                            clear: function(e) {
                                var t = this
                                  , r = t.ready().then(function() {
                                    for (var e, r = t._dbInfo.keyPrefix, n = localStorage.length - 1; 0 <= n; n--)
                                        0 === (e = localStorage.key(n)).indexOf(r) && localStorage.removeItem(e)
                                });
                                return n(r, e),
                                r
                            },
                            length: function(e) {
                                var t = this.keys().then(function(e) {
                                    return e.length
                                });
                                return n(t, e),
                                t
                            },
                            key: function(e, t) {
                                var r = this
                                  , i = r.ready().then(function() {
                                    var t, n = r._dbInfo;
                                    try {
                                        t = localStorage.key(e)
                                    } catch (e) {
                                        t = null
                                    }
                                    return t && (t = t.substring(n.keyPrefix.length)),
                                    t
                                });
                                return n(i, t),
                                i
                            },
                            keys: function(e) {
                                var t = this
                                  , r = t.ready().then(function() {
                                    for (var e, r = t._dbInfo, n = localStorage.length, i = [], a = 0; a < n; a++)
                                        0 === (e = localStorage.key(a)).indexOf(r.keyPrefix) && i.push(e.substring(r.keyPrefix.length));
                                    return i
                                });
                                return n(r, e),
                                r
                            },
                            dropInstance: function(e, t) {
                                if (t = o.apply(this, arguments),
                                !(e = "function" != typeof e && e || {}).name) {
                                    var r = this.config();
                                    e.name = e.name || r.name,
                                    e.storeName = e.storeName || r.storeName
                                }
                                var i, a = this;
                                return n(i = e.name ? new C(function(t) {
                                    e.storeName ? t(k(e, a._defaultConfig)) : t(e.name + "/")
                                }
                                ).then(function(e) {
                                    for (var t, r = localStorage.length - 1; 0 <= r; r--)
                                        0 === (t = localStorage.key(r)).indexOf(e) && localStorage.removeItem(t)
                                }) : C.reject("Invalid arguments"), t),
                                i
                            }
                        }
                          , Z = function(e, t) {
                            return e === t || "number" == typeof e && "number" == typeof t && isNaN(e) && isNaN(t)
                        }
                          , K = function(e, t) {
                            for (var r = e.length, n = 0; n < r; ) {
                                if (Z(e[n], t))
                                    return !0;
                                n++
                            }
                            return !1
                        }
                          , q = Array.isArray || function(e) {
                            return "[object Array]" === Object.prototype.toString.call(e)
                        }
                          , Q = {}
                          , J = {}
                          , ee = {
                            INDEXEDDB: O,
                            WEBSQL: X,
                            LOCALSTORAGE: Y
                        }
                          , te = [ee.INDEXEDDB._driver, ee.WEBSQL._driver, ee.LOCALSTORAGE._driver]
                          , re = ["dropInstance"]
                          , ne = ["clear", "getItem", "iterate", "key", "keys", "length", "removeItem", "setItem"].concat(re)
                          , ie = {
                            description: "",
                            driver: te.slice(),
                            name: "localforage",
                            size: 4980736,
                            storeName: "keyvaluepairs",
                            version: 1
                        }
                          , ae = new (function() {
                            function e(t) {
                                for (var r in function(e, t) {
                                    if (!(e instanceof t))
                                        throw new TypeError("Cannot call a class as a function")
                                }(this, e),
                                ee)
                                    if (ee.hasOwnProperty(r)) {
                                        var n = ee[r]
                                          , i = n._driver;
                                        this[r] = i,
                                        Q[i] || this.defineDriver(n)
                                    }
                                this._defaultConfig = S({}, ie),
                                this._config = S({}, this._defaultConfig, t),
                                this._driverSet = null,
                                this._initDriver = null,
                                this._ready = !1,
                                this._dbInfo = null,
                                this._wrapLibraryMethodsWithReady(),
                                this.setDriver(this._config.driver).catch(function() {})
                            }
                            return e.prototype.config = function(e) {
                                if ("object" === (void 0 === e ? "undefined" : T(e))) {
                                    if (this._ready)
                                        return new Error("Can't call config() after localforage has been used.");
                                    for (var t in e) {
                                        if ("storeName" == t && (e[t] = e[t].replace(/\W/g, "_")),
                                        "version" == t && "number" != typeof e[t])
                                            return new Error("Database version must be a number.");
                                        this._config[t] = e[t]
                                    }
                                    return !("driver"in e && e.driver) || this.setDriver(this._config.driver)
                                }
                                return "string" == typeof e ? this._config[e] : this._config
                            }
                            ,
                            e.prototype.defineDriver = function(e, t, r) {
                                var a = new C(function(t, r) {
                                    try {
                                        var i = e._driver
                                          , a = new Error("Custom driver not compliant; see https://mozilla.github.io/localForage/#definedriver");
                                        if (!e._driver)
                                            return void r(a);
                                        for (var o = ne.concat("_initStorage"), s = 0, l = o.length; s < l; s++) {
                                            var c = o[s];
                                            if ((!K(re, c) || e[c]) && "function" != typeof e[c])
                                                return void r(a)
                                        }
                                        !function() {
                                            for (var t, r = function(e) {
                                                return function() {
                                                    var t = new Error("Method " + e + " is not implemented by the current driver")
                                                      , r = C.reject(t);
                                                    return n(r, arguments[arguments.length - 1]),
                                                    r
                                                }
                                            }, i = 0, a = re.length; i < a; i++)
                                                e[t = re[i]] || (e[t] = r(t))
                                        }();
                                        var u = function(r) {
                                            Q[i] && console.info("Redefining LocalForage driver: " + i),
                                            Q[i] = e,
                                            J[i] = r,
                                            t()
                                        };
                                        "_support"in e ? e._support && "function" == typeof e._support ? e._support().then(u, r) : u(!!e._support) : u(!0)
                                    } catch (t) {
                                        r(t)
                                    }
                                }
                                );
                                return i(a, t, r),
                                a
                            }
                            ,
                            e.prototype.driver = function() {
                                return this._driver || null
                            }
                            ,
                            e.prototype.getDriver = function(e, t, r) {
                                var n = Q[e] ? C.resolve(Q[e]) : C.reject(new Error("Driver not found."));
                                return i(n, t, r),
                                n
                            }
                            ,
                            e.prototype.getSerializer = function(e) {
                                var t = C.resolve(G);
                                return i(t, e),
                                t
                            }
                            ,
                            e.prototype.ready = function(e) {
                                var t = this
                                  , r = t._driverSet.then(function() {
                                    return null === t._ready && (t._ready = t._initDriver()),
                                    t._ready
                                });
                                return i(r, e, e),
                                r
                            }
                            ,
                            e.prototype.setDriver = function(e, t, r) {
                                function n() {
                                    o._config.driver = o.driver()
                                }
                                function a(e) {
                                    return o._extend(e),
                                    n(),
                                    o._ready = o._initStorage(o._config),
                                    o._ready
                                }
                                var o = this;
                                q(e) || (e = [e]);
                                var s = this._getSupportedDrivers(e)
                                  , l = null === this._driverSet ? C.resolve() : this._driverSet.catch(function() {
                                    return C.resolve()
                                });
                                return this._driverSet = l.then(function() {
                                    var e = s[0];
                                    return o._dbInfo = null,
                                    o._ready = null,
                                    o.getDriver(e).then(function(e) {
                                        o._driver = e._driver,
                                        n(),
                                        o._wrapLibraryMethodsWithReady(),
                                        o._initDriver = function(e) {
                                            return function() {
                                                var t = 0;
                                                return function r() {
                                                    for (; t < e.length; ) {
                                                        var i = e[t];
                                                        return t++,
                                                        o._dbInfo = null,
                                                        o._ready = null,
                                                        o.getDriver(i).then(a).catch(r)
                                                    }
                                                    n();
                                                    var s = new Error("No available storage method found.");
                                                    return o._driverSet = C.reject(s),
                                                    o._driverSet
                                                }()
                                            }
                                        }(s)
                                    })
                                }).catch(function() {
                                    n();
                                    var e = new Error("No available storage method found.");
                                    return o._driverSet = C.reject(e),
                                    o._driverSet
                                }),
                                i(this._driverSet, t, r),
                                this._driverSet
                            }
                            ,
                            e.prototype.supports = function(e) {
                                return !!J[e]
                            }
                            ,
                            e.prototype._extend = function(e) {
                                S(this, e)
                            }
                            ,
                            e.prototype._getSupportedDrivers = function(e) {
                                for (var t, r = [], n = 0, i = e.length; n < i; n++)
                                    t = e[n],
                                    this.supports(t) && r.push(t);
                                return r
                            }
                            ,
                            e.prototype._wrapLibraryMethodsWithReady = function() {
                                for (var e = 0, t = ne.length; e < t; e++)
                                    A(this, ne[e])
                            }
                            ,
                            e.prototype.createInstance = function(t) {
                                return new e(t)
                            }
                            ,
                            e
                        }());
                        t.exports = ae
                    }
                    , {
                        3: 3
                    }]
                }, {}, [4])(4)
            })
        }
        ).call(this, r(10))
    }
    , function(e, t) {
        "use strict";
        function r(e, t) {
            return Object.prototype.hasOwnProperty.call(e, t)
        }
        var n = "undefined" != typeof Uint8Array && "undefined" != typeof Uint16Array && "undefined" != typeof Int32Array;
        t.assign = function(e) {
            for (var t = Array.prototype.slice.call(arguments, 1); t.length; ) {
                var n = t.shift();
                if (n) {
                    if ("object" != typeof n)
                        throw new TypeError(n + "must be non-object");
                    for (var i in n)
                        r(n, i) && (e[i] = n[i])
                }
            }
            return e
        }
        ,
        t.shrinkBuf = function(e, t) {
            return e.length === t ? e : e.subarray ? e.subarray(0, t) : (e.length = t,
            e)
        }
        ;
        var i = {
            arraySet: function(e, t, r, n, i) {
                if (t.subarray && e.subarray)
                    e.set(t.subarray(r, r + n), i);
                else
                    for (var a = 0; a < n; a++)
                        e[i + a] = t[r + a]
            },
            flattenChunks: function(e) {
                var t, r, n, i, a, o;
                for (n = 0,
                t = 0,
                r = e.length; t < r; t++)
                    n += e[t].length;
                for (o = new Uint8Array(n),
                i = 0,
                t = 0,
                r = e.length; t < r; t++)
                    a = e[t],
                    o.set(a, i),
                    i += a.length;
                return o
            }
        }
          , a = {
            arraySet: function(e, t, r, n, i) {
                for (var a = 0; a < n; a++)
                    e[i + a] = t[r + a]
            },
            flattenChunks: function(e) {
                return [].concat.apply([], e)
            }
        };
        t.setTyped = function(e) {
            e ? (t.Buf8 = Uint8Array,
            t.Buf16 = Uint16Array,
            t.Buf32 = Int32Array,
            t.assign(t, i)) : (t.Buf8 = Array,
            t.Buf16 = Array,
            t.Buf32 = Array,
            t.assign(t, a))
        }
        ,
        t.setTyped(n)
    }
    , function(module, __webpack_exports__, __webpack_require__) {
        "use strict";
        __webpack_require__.d(__webpack_exports__, "a", function() {
            return PlaceholderContext
        });
        class PlaceholderContext {
            constructor(e) {
                this.values = {},
                e && this.update(e)
            }
            get isResolved() {
                return Object.values(this.values).every(e=>"number" == typeof e)
            }
            update(e) {
                this.values = Object.assign(this.values, e)
            }
            resolve(placeholder) {
                if ("object" != typeof placeholder)
                    return placeholder;
                if (1 == Object.keys(placeholder).length && "eval"in placeholder) {
                    if (!this.isResolved)
                        throw Error(`Not all placeholders are resolved: ${this}`);
                    return eval("(function(placeholders){return " + placeholder.eval + ";})")(this.values)
                }
                return placeholder instanceof Array ? placeholder.map(e=>this.resolve(e)) : Object.entries(placeholder).reduce((e,[t,r])=>(e[t] = this.resolve(r),
                e), {})
            }
            toString() {
                return JSON.stringify(this.values)
            }
        }
    }
    , function(e) {
        "use strict";
        e.exports = {
            2: "need dictionary",
            1: "stream end",
            0: "",
            "-1": "file error",
            "-2": "stream error",
            "-3": "data error",
            "-4": "insufficient memory",
            "-5": "buffer error",
            "-6": "incompatible version"
        }
    }
    , function(e) {
        "use strict";
        e.exports = function(e, t, r, n) {
            for (var i = 0 | 65535 & e, a = 0 | 65535 & e >>> 16, o = 0; 0 !== r; ) {
                r -= o = 2e3 < r ? 2e3 : r;
                do {
                    a = 0 | a + (i = 0 | i + t[n++])
                } while (--o);
                i %= 65521,
                a %= 65521
            }
            return i | a << 16 | 0
        }
    }
    , function(e) {
        "use strict";
        var t = function() {
            for (var e, t = [], r = 0; 256 > r; r++) {
                e = r;
                for (var n = 0; 8 > n; n++)
                    e = 1 & e ? 3988292384 ^ e >>> 1 : e >>> 1;
                t[r] = e
            }
            return t
        }();
        e.exports = function(e, r, n, i) {
            e ^= -1;
            for (var a = i; a < i + n; a++)
                e = e >>> 8 ^ t[255 & (e ^ r[a])];
            return -1 ^ e
        }
    }
    , function(e, t, r) {
        "use strict";
        function n(e, t) {
            if (65534 > t && (e.subarray && o || !e.subarray && a))
                return _StringfromCharCode.apply(null, i.shrinkBuf(e, t));
            for (var r = "", n = 0; n < t; n++)
                r += _StringfromCharCode(e[n]);
            return r
        }
        var i = r(1)
          , a = !0
          , o = !0;
        try {
            _StringfromCharCode.apply(null, [0])
        } catch (e) {
            a = !1
        }
        try {
            _StringfromCharCode.apply(null, new Uint8Array(1))
        } catch (e) {
            o = !1
        }
        for (var s = new i.Buf8(256), l = 0; 256 > l; l++)
            s[l] = 252 <= l ? 6 : 248 <= l ? 5 : 240 <= l ? 4 : 224 <= l ? 3 : 192 <= l ? 2 : 1;
        s[254] = s[254] = 1,
        t.string2buf = function(e) {
            var t, r, n, a, o, s = e.length, l = 0;
            for (a = 0; a < s; a++)
                55296 == (64512 & (r = e.charCodeAt(a))) && a + 1 < s && (56320 == (64512 & (n = e.charCodeAt(a + 1))) && (r = 65536 + (r - 55296 << 10) + (n - 56320),
                a++)),
                l += 128 > r ? 1 : 2048 > r ? 2 : 65536 > r ? 3 : 4;
            for (t = new i.Buf8(l),
            o = 0,
            a = 0; o < l; a++)
                55296 == (64512 & (r = e.charCodeAt(a))) && a + 1 < s && (56320 == (64512 & (n = e.charCodeAt(a + 1))) && (r = 65536 + (r - 55296 << 10) + (n - 56320),
                a++)),
                128 > r ? t[o++] = r : 2048 > r ? (t[o++] = 192 | r >>> 6,
                t[o++] = 128 | 63 & r) : 65536 > r ? (t[o++] = 224 | r >>> 12,
                t[o++] = 128 | 63 & r >>> 6,
                t[o++] = 128 | 63 & r) : (t[o++] = 240 | r >>> 18,
                t[o++] = 128 | 63 & r >>> 12,
                t[o++] = 128 | 63 & r >>> 6,
                t[o++] = 128 | 63 & r);
            return t
        }
        ,
        t.buf2binstring = function(e) {
            return n(e, e.length)
        }
        ,
        t.binstring2buf = function(e) {
            for (var t = new i.Buf8(e.length), r = 0, n = t.length; r < n; r++)
                t[r] = e.charCodeAt(r);
            return t
        }
        ,
        t.buf2string = function(e, t) {
            var r, i, a, o, l = t || e.length, c = Array(2 * l);
            for (i = 0,
            r = 0; r < l; )
                if (128 > (a = e[r++]))
                    c[i++] = a;
                else if (4 < (o = s[a]))
                    c[i++] = 65533,
                    r += o - 1;
                else {
                    for (a &= 2 === o ? 31 : 3 === o ? 15 : 7; 1 < o && r < l; )
                        a = a << 6 | 63 & e[r++],
                        o--;
                    1 < o ? c[i++] = 65533 : 65536 > a ? c[i++] = a : (a -= 65536,
                    c[i++] = 55296 | 1023 & a >> 10,
                    c[i++] = 56320 | 1023 & a)
                }
            return n(c, i)
        }
        ,
        t.utf8border = function(e, t) {
            var r;
            for ((t = t || e.length) > e.length && (t = e.length),
            r = t - 1; 0 <= r && 128 == (192 & e[r]); )
                r--;
            return 0 > r ? t : 0 === r ? t : r + s[e[r]] > t ? r : t
        }
    }
    , function(e) {
        "use strict";
        e.exports = function() {
            this.input = null,
            this.next_in = 0,
            this.avail_in = 0,
            this.total_in = 0,
            this.output = null,
            this.next_out = 0,
            this.avail_out = 0,
            this.total_out = 0,
            this.msg = "",
            this.state = null,
            this.data_type = 2,
            this.adler = 0
        }
    }
    , function(e) {
        "use strict";
        e.exports = {
            Z_NO_FLUSH: 0,
            Z_PARTIAL_FLUSH: 1,
            Z_SYNC_FLUSH: 2,
            Z_FULL_FLUSH: 3,
            Z_FINISH: 4,
            Z_BLOCK: 5,
            Z_TREES: 6,
            Z_OK: 0,
            Z_STREAM_END: 1,
            Z_NEED_DICT: 2,
            Z_ERRNO: -1,
            Z_STREAM_ERROR: -2,
            Z_DATA_ERROR: -3,
            Z_BUF_ERROR: -5,
            Z_NO_COMPRESSION: 0,
            Z_BEST_SPEED: 1,
            Z_BEST_COMPRESSION: 9,
            Z_DEFAULT_COMPRESSION: -1,
            Z_FILTERED: 1,
            Z_HUFFMAN_ONLY: 2,
            Z_RLE: 3,
            Z_FIXED: 4,
            Z_DEFAULT_STRATEGY: 0,
            Z_BINARY: 0,
            Z_TEXT: 1,
            Z_UNKNOWN: 2,
            Z_DEFLATED: 8
        }
    }
    , function(e, t, r) {
        "use strict";
        var n = {};
        (0,
        r(1).assign)(n, r(11), r(14), r(8)),
        e.exports = n
    }
    , function(e) {
        var t = function() {
            return this
        }();
        try {
            t = t || new Function("return this")()
        } catch (e) {
            "object" == typeof window && (t = window)
        }
        e.exports = t
    }
    , function(e, t, r) {
        "use strict";
        function n(e) {
            if (!(this instanceof n))
                return new n(e);
            this.options = o.assign({
                level: f,
                method: p,
                chunkSize: 16384,
                windowBits: 15,
                memLevel: 8,
                strategy: d,
                to: ""
            }, e || {});
            var t = this.options;
            t.raw && 0 < t.windowBits ? t.windowBits = -t.windowBits : t.gzip && 0 < t.windowBits && 16 > t.windowBits && (t.windowBits += 16),
            this.err = 0,
            this.msg = "",
            this.ended = !1,
            this.chunks = [],
            this.strm = new c,
            this.strm.avail_out = 0;
            var r = a.deflateInit2(this.strm, t.level, t.method, t.windowBits, t.memLevel, t.strategy);
            if (r !== h)
                throw new Error(l[r]);
            if (t.header && a.deflateSetHeader(this.strm, t.header),
            t.dictionary) {
                var i;
                if (i = "string" == typeof t.dictionary ? s.string2buf(t.dictionary) : "[object ArrayBuffer]" === u.call(t.dictionary) ? new Uint8Array(t.dictionary) : t.dictionary,
                (r = a.deflateSetDictionary(this.strm, i)) !== h)
                    throw new Error(l[r]);
                this._dict_set = !0
            }
        }
        function i(e, t) {
            var r = new n(t);
            if (r.push(e, !0),
            r.err)
                throw r.msg || l[r.err];
            return r.result
        }
        var a = r(12)
          , o = r(1)
          , s = r(6)
          , l = r(3)
          , c = r(7)
          , u = Object.prototype.toString
          , h = 0
          , f = -1
          , d = 0
          , p = 8;
        n.prototype.push = function(e, t) {
            var r, n, i = this.strm, l = this.options.chunkSize;
            if (this.ended)
                return !1;
            n = t === ~~t ? t : !0 === t ? 4 : 0,
            i.input = "string" == typeof e ? s.string2buf(e) : "[object ArrayBuffer]" === u.call(e) ? new Uint8Array(e) : e,
            i.next_in = 0,
            i.avail_in = i.input.length;
            do {
                if (0 === i.avail_out && (i.output = new o.Buf8(l),
                i.next_out = 0,
                i.avail_out = l),
                1 !== (r = a.deflate(i, n)) && r !== h)
                    return this.onEnd(r),
                    this.ended = !0,
                    !1;
                (0 === i.avail_out || 0 === i.avail_in && (4 === n || 2 === n)) && ("string" === this.options.to ? this.onData(s.buf2binstring(o.shrinkBuf(i.output, i.next_out))) : this.onData(o.shrinkBuf(i.output, i.next_out)))
            } while ((0 < i.avail_in || 0 === i.avail_out) && 1 !== r);
            return 4 === n ? (r = a.deflateEnd(this.strm),
            this.onEnd(r),
            this.ended = !0,
            r === h) : 2 !== n || (this.onEnd(h),
            i.avail_out = 0,
            !0)
        }
        ,
        n.prototype.onData = function(e) {
            this.chunks.push(e)
        }
        ,
        n.prototype.onEnd = function(e) {
            e === h && ("string" === this.options.to ? this.result = this.chunks.join("") : this.result = o.flattenChunks(this.chunks)),
            this.chunks = [],
            this.err = e,
            this.msg = this.strm.msg
        }
        ,
        t.Deflate = n,
        t.deflate = i,
        t.deflateRaw = function(e, t) {
            return (t = t || {}).raw = !0,
            i(e, t)
        }
        ,
        t.gzip = function(e, t) {
            return (t = t || {}).gzip = !0,
            i(e, t)
        }
    }
    , function(e, t, r) {
        "use strict";
        function n(e, t) {
            return e.msg = T[t],
            t
        }
        function i(e) {
            return (e << 1) - (4 < e ? 9 : 0)
        }
        function a(e) {
            for (var t = e.length; 0 <= --t; )
                e[t] = 0
        }
        function o(e) {
            var t = e.state
              , r = t.pending;
            r > e.avail_out && (r = e.avail_out),
            0 === r || (k.arraySet(e.output, t.pending_buf, t.pending_out, r, e.next_out),
            e.next_out += r,
            t.pending_out += r,
            e.total_out += r,
            e.avail_out -= r,
            t.pending -= r,
            0 === t.pending && (t.pending_out = 0))
        }
        function s(e, t) {
            x._tr_flush_block(e, 0 <= e.block_start ? e.block_start : -1, e.strstart - e.block_start, t),
            e.block_start = e.strstart,
            o(e.strm)
        }
        function l(e, t) {
            e.pending_buf[e.pending++] = t
        }
        function c(e, t) {
            e.pending_buf[e.pending++] = 255 & t >>> 8,
            e.pending_buf[e.pending++] = 255 & t
        }
        function u(e, t, r, n) {
            var i = e.avail_in;
            return i > n && (i = n),
            0 === i ? 0 : (e.avail_in -= i,
            k.arraySet(t, e.input, e.next_in, i, r),
            1 === e.state.wrap ? e.adler = A(e.adler, t, i, r) : 2 === e.state.wrap && (e.adler = S(e.adler, t, i, r)),
            e.next_in += i,
            e.total_in += i,
            i)
        }
        function h(e, t) {
            var r, n, i = e.max_chain_length, a = e.strstart, o = e.prev_length, s = e.nice_match, l = e.strstart > e.w_size - Q ? e.strstart - (e.w_size - Q) : 0, c = e.window, u = e.w_mask, h = e.prev, f = e.strstart + q, d = c[a + o - 1], p = c[a + o];
            e.prev_length >= e.good_match && (i >>= 2),
            s > e.lookahead && (s = e.lookahead);
            do {
                if (c[(r = t) + o] === p && c[r + o - 1] === d && c[r] === c[a] && c[++r] === c[a + 1]) {
                    a += 2,
                    r++;
                    do {} while (c[++a] === c[++r] && c[++a] === c[++r] && c[++a] === c[++r] && c[++a] === c[++r] && c[++a] === c[++r] && c[++a] === c[++r] && c[++a] === c[++r] && c[++a] === c[++r] && a < f);
                    if (n = q - (f - a),
                    a = f - q,
                    n > o) {
                        if (e.match_start = t,
                        o = n,
                        n >= s)
                            break;
                        d = c[a + o - 1],
                        p = c[a + o]
                    }
                }
            } while ((t = h[t & u]) > l && 0 != --i);
            return o <= e.lookahead ? o : e.lookahead
        }
        function f(e) {
            var t, r, n, i, a, o = e.w_size;
            do {
                if (i = e.window_size - e.lookahead - e.strstart,
                e.strstart >= o + (o - Q)) {
                    k.arraySet(e.window, e.window, o, o, 0),
                    e.match_start -= o,
                    e.strstart -= o,
                    e.block_start -= o,
                    t = r = e.hash_size;
                    do {
                        n = e.head[--t],
                        e.head[t] = n >= o ? n - o : 0
                    } while (--r);
                    t = r = o;
                    do {
                        n = e.prev[--t],
                        e.prev[t] = n >= o ? n - o : 0
                    } while (--r);
                    i += o
                }
                if (0 === e.strm.avail_in)
                    break;
                if (r = u(e.strm, e.window, e.strstart + e.lookahead, i),
                e.lookahead += r,
                e.lookahead + e.insert >= K)
                    for (a = e.strstart - e.insert,
                    e.ins_h = e.window[a],
                    e.ins_h = (e.ins_h << e.hash_shift ^ e.window[a + 1]) & e.hash_mask; e.insert && (e.ins_h = (e.ins_h << e.hash_shift ^ e.window[a + K - 1]) & e.hash_mask,
                    e.prev[a & e.w_mask] = e.head[e.ins_h],
                    e.head[e.ins_h] = a,
                    a++,
                    e.insert--,
                    !(e.lookahead + e.insert < K)); )
                        ;
            } while (e.lookahead < Q && 0 !== e.strm.avail_in)
        }
        function d(e, t) {
            for (var r, n; ; ) {
                if (e.lookahead < Q) {
                    if (f(e),
                    e.lookahead < Q && t === I)
                        return se;
                    if (0 === e.lookahead)
                        break
                }
                if (r = 0,
                e.lookahead >= K && (e.ins_h = (e.ins_h << e.hash_shift ^ e.window[e.strstart + K - 1]) & e.hash_mask,
                r = e.prev[e.strstart & e.w_mask] = e.head[e.ins_h],
                e.head[e.ins_h] = e.strstart),
                0 !== r && e.strstart - r <= e.w_size - Q && (e.match_length = h(e, r)),
                e.match_length >= K)
                    if (n = x._tr_tally(e, e.strstart - e.match_start, e.match_length - K),
                    e.lookahead -= e.match_length,
                    e.match_length <= e.max_lazy_match && e.lookahead >= K) {
                        e.match_length--;
                        do {
                            e.strstart++,
                            e.ins_h = (e.ins_h << e.hash_shift ^ e.window[e.strstart + K - 1]) & e.hash_mask,
                            r = e.prev[e.strstart & e.w_mask] = e.head[e.ins_h],
                            e.head[e.ins_h] = e.strstart
                        } while (0 != --e.match_length);
                        e.strstart++
                    } else
                        e.strstart += e.match_length,
                        e.match_length = 0,
                        e.ins_h = e.window[e.strstart],
                        e.ins_h = (e.ins_h << e.hash_shift ^ e.window[e.strstart + 1]) & e.hash_mask;
                else
                    n = x._tr_tally(e, 0, e.window[e.strstart]),
                    e.lookahead--,
                    e.strstart++;
                if (n && (s(e, !1),
                0 === e.strm.avail_out))
                    return se
            }
            return e.insert = e.strstart < K - 1 ? e.strstart : K - 1,
            t === R ? (s(e, !0),
            0 === e.strm.avail_out ? ce : ue) : e.last_lit && (s(e, !1),
            0 === e.strm.avail_out) ? se : le
        }
        function p(e, t) {
            for (var r, n, i; ; ) {
                if (e.lookahead < Q) {
                    if (f(e),
                    e.lookahead < Q && t === I)
                        return se;
                    if (0 === e.lookahead)
                        break
                }
                if (r = 0,
                e.lookahead >= K && (e.ins_h = (e.ins_h << e.hash_shift ^ e.window[e.strstart + K - 1]) & e.hash_mask,
                r = e.prev[e.strstart & e.w_mask] = e.head[e.ins_h],
                e.head[e.ins_h] = e.strstart),
                e.prev_length = e.match_length,
                e.prev_match = e.match_start,
                e.match_length = K - 1,
                0 !== r && e.prev_length < e.max_lazy_match && e.strstart - r <= e.w_size - Q && (e.match_length = h(e, r),
                5 >= e.match_length && (e.strategy === L || e.match_length === K && 4096 < e.strstart - e.match_start) && (e.match_length = K - 1)),
                e.prev_length >= K && e.match_length <= e.prev_length) {
                    i = e.strstart + e.lookahead - K,
                    n = x._tr_tally(e, e.strstart - 1 - e.prev_match, e.prev_length - K),
                    e.lookahead -= e.prev_length - 1,
                    e.prev_length -= 2;
                    do {
                        ++e.strstart <= i && (e.ins_h = (e.ins_h << e.hash_shift ^ e.window[e.strstart + K - 1]) & e.hash_mask,
                        r = e.prev[e.strstart & e.w_mask] = e.head[e.ins_h],
                        e.head[e.ins_h] = e.strstart)
                    } while (0 != --e.prev_length);
                    if (e.match_available = 0,
                    e.match_length = K - 1,
                    e.strstart++,
                    n && (s(e, !1),
                    0 === e.strm.avail_out))
                        return se
                } else if (e.match_available) {
                    if ((n = x._tr_tally(e, 0, e.window[e.strstart - 1])) && s(e, !1),
                    e.strstart++,
                    e.lookahead--,
                    0 === e.strm.avail_out)
                        return se
                } else
                    e.match_available = 1,
                    e.strstart++,
                    e.lookahead--
            }
            return e.match_available && (n = x._tr_tally(e, 0, e.window[e.strstart - 1]),
            e.match_available = 0),
            e.insert = e.strstart < K - 1 ? e.strstart : K - 1,
            t === R ? (s(e, !0),
            0 === e.strm.avail_out ? ce : ue) : e.last_lit && (s(e, !1),
            0 === e.strm.avail_out) ? se : le
        }
        function _(e, t) {
            for (var r, n, i, a, o = e.window; ; ) {
                if (e.lookahead <= q) {
                    if (f(e),
                    e.lookahead <= q && t === I)
                        return se;
                    if (0 === e.lookahead)
                        break
                }
                if (e.match_length = 0,
                e.lookahead >= K && 0 < e.strstart && ((n = o[i = e.strstart - 1]) === o[++i] && n === o[++i] && n === o[++i])) {
                    a = e.strstart + q;
                    do {} while (n === o[++i] && n === o[++i] && n === o[++i] && n === o[++i] && n === o[++i] && n === o[++i] && n === o[++i] && n === o[++i] && i < a);
                    e.match_length = q - (a - i),
                    e.match_length > e.lookahead && (e.match_length = e.lookahead)
                }
                if (e.match_length >= K ? (r = x._tr_tally(e, 1, e.match_length - K),
                e.lookahead -= e.match_length,
                e.strstart += e.match_length,
                e.match_length = 0) : (r = x._tr_tally(e, 0, e.window[e.strstart]),
                e.lookahead--,
                e.strstart++),
                r && (s(e, !1),
                0 === e.strm.avail_out))
                    return se
            }
            return e.insert = 0,
            t === R ? (s(e, !0),
            0 === e.strm.avail_out ? ce : ue) : e.last_lit && (s(e, !1),
            0 === e.strm.avail_out) ? se : le
        }
        function m(e, t) {
            for (var r; ; ) {
                if (0 === e.lookahead && (f(e),
                0 === e.lookahead)) {
                    if (t === I)
                        return se;
                    break
                }
                if (e.match_length = 0,
                r = x._tr_tally(e, 0, e.window[e.strstart]),
                e.lookahead--,
                e.strstart++,
                r && (s(e, !1),
                0 === e.strm.avail_out))
                    return se
            }
            return e.insert = 0,
            t === R ? (s(e, !0),
            0 === e.strm.avail_out ? ce : ue) : e.last_lit && (s(e, !1),
            0 === e.strm.avail_out) ? se : le
        }
        function w(e, t, r, n, i) {
            this.good_length = e,
            this.max_lazy = t,
            this.nice_length = r,
            this.max_chain = n,
            this.func = i
        }
        function b() {
            this.strm = null,
            this.status = 0,
            this.pending_buf = null,
            this.pending_buf_size = 0,
            this.pending_out = 0,
            this.pending = 0,
            this.wrap = 0,
            this.gzhead = null,
            this.gzindex = 0,
            this.method = H,
            this.last_flush = -1,
            this.w_size = 0,
            this.w_bits = 0,
            this.w_mask = 0,
            this.window = null,
            this.window_size = 0,
            this.prev = null,
            this.head = null,
            this.ins_h = 0,
            this.hash_size = 0,
            this.hash_bits = 0,
            this.hash_mask = 0,
            this.hash_shift = 0,
            this.block_start = 0,
            this.match_length = 0,
            this.prev_match = 0,
            this.match_available = 0,
            this.strstart = 0,
            this.match_start = 0,
            this.lookahead = 0,
            this.prev_length = 0,
            this.max_chain_length = 0,
            this.max_lazy_match = 0,
            this.level = 0,
            this.strategy = 0,
            this.good_match = 0,
            this.nice_match = 0,
            this.dyn_ltree = new k.Buf16(2 * Y),
            this.dyn_dtree = new k.Buf16(2 * (2 * G + 1)),
            this.bl_tree = new k.Buf16(2 * (2 * X + 1)),
            a(this.dyn_ltree),
            a(this.dyn_dtree),
            a(this.bl_tree),
            this.l_desc = null,
            this.d_desc = null,
            this.bl_desc = null,
            this.bl_count = new k.Buf16(Z + 1),
            this.heap = new k.Buf16(2 * $ + 1),
            a(this.heap),
            this.heap_len = 0,
            this.heap_max = 0,
            this.depth = new k.Buf16(2 * $ + 1),
            a(this.depth),
            this.l_buf = 0,
            this.lit_bufsize = 0,
            this.last_lit = 0,
            this.d_buf = 0,
            this.opt_len = 0,
            this.static_len = 0,
            this.matches = 0,
            this.insert = 0,
            this.bi_buf = 0,
            this.bi_valid = 0
        }
        function g(e) {
            var t;
            return e && e.state ? (e.total_in = e.total_out = 0,
            e.data_type = W,
            (t = e.state).pending = 0,
            t.pending_out = 0,
            0 > t.wrap && (t.wrap = -t.wrap),
            t.status = t.wrap ? ee : ae,
            e.adler = 2 === t.wrap ? 0 : 1,
            t.last_flush = I,
            x._tr_init(t),
            z) : n(e, P)
        }
        function y(e) {
            var t = g(e);
            return t === z && function(e) {
                e.window_size = 2 * e.w_size,
                a(e.head),
                e.max_lazy_match = E[e.level].max_lazy,
                e.good_match = E[e.level].good_length,
                e.nice_match = E[e.level].nice_length,
                e.max_chain_length = E[e.level].max_chain,
                e.strstart = 0,
                e.block_start = 0,
                e.lookahead = 0,
                e.insert = 0,
                e.match_length = e.prev_length = K - 1,
                e.match_available = 0,
                e.ins_h = 0
            }(e.state),
            t
        }
        function v(e, t, r, i, a, o) {
            if (!e)
                return P;
            var s = 1;
            if (t === F && (t = 6),
            0 > i ? (s = 0,
            i = -i) : 15 < i && (s = 2,
            i -= 16),
            1 > a || a > V || r !== H || 8 > i || 15 < i || 0 > t || 9 < t || 0 > o || o > j)
                return n(e, P);
            8 === i && (i = 9);
            var l = new b;
            return e.state = l,
            l.strm = e,
            l.wrap = s,
            l.gzhead = null,
            l.w_bits = i,
            l.w_size = 1 << l.w_bits,
            l.w_mask = l.w_size - 1,
            l.hash_bits = a + 7,
            l.hash_size = 1 << l.hash_bits,
            l.hash_mask = l.hash_size - 1,
            l.hash_shift = ~~((l.hash_bits + K - 1) / K),
            l.window = new k.Buf8(2 * l.w_size),
            l.head = new k.Buf16(l.hash_size),
            l.prev = new k.Buf16(l.w_size),
            l.lit_bufsize = 1 << a + 6,
            l.pending_buf_size = 4 * l.lit_bufsize,
            l.pending_buf = new k.Buf8(l.pending_buf_size),
            l.d_buf = 1 * l.lit_bufsize,
            l.l_buf = 3 * l.lit_bufsize,
            l.level = t,
            l.strategy = o,
            l.method = r,
            y(e)
        }
        var E, k = r(1), x = r(13), A = r(4), S = r(5), T = r(3), I = 0, C = 1, B = 3, R = 4, D = 5, z = 0, N = 1, P = -2, O = -5, F = -1, L = 1, U = 2, M = 3, j = 4, W = 2, H = 8, V = 9, $ = 286, G = 30, X = 19, Y = 2 * $ + 1, Z = 15, K = 3, q = 258, Q = q + K + 1, J = 32, ee = 42, te = 69, re = 73, ne = 91, ie = 103, ae = 113, oe = 666, se = 1, le = 2, ce = 3, ue = 4, he = 3;
        E = [new w(0,0,0,0,function(e, t) {
            var r = 65535;
            for (r > e.pending_buf_size - 5 && (r = e.pending_buf_size - 5); ; ) {
                if (1 >= e.lookahead) {
                    if (f(e),
                    0 === e.lookahead && t === I)
                        return se;
                    if (0 === e.lookahead)
                        break
                }
                e.strstart += e.lookahead,
                e.lookahead = 0;
                var n = e.block_start + r;
                if ((0 === e.strstart || e.strstart >= n) && (e.lookahead = e.strstart - n,
                e.strstart = n,
                s(e, !1),
                0 === e.strm.avail_out))
                    return se;
                if (e.strstart - e.block_start >= e.w_size - Q && (s(e, !1),
                0 === e.strm.avail_out))
                    return se
            }
            return e.insert = 0,
            t === R ? (s(e, !0),
            0 === e.strm.avail_out ? ce : ue) : (e.strstart > e.block_start && (s(e, !1),
            e.strm.avail_out),
            se)
        }
        ), new w(4,4,8,4,d), new w(4,5,16,8,d), new w(4,6,32,32,d), new w(4,4,16,16,p), new w(8,16,32,32,p), new w(8,16,128,128,p), new w(8,32,128,256,p), new w(32,128,258,1024,p), new w(32,258,258,4096,p)],
        t.deflateInit = function(e, t) {
            return v(e, t, H, 15, 8, 0)
        }
        ,
        t.deflateInit2 = v,
        t.deflateReset = y,
        t.deflateResetKeep = g,
        t.deflateSetHeader = function(e, t) {
            return e && e.state && 2 === e.state.wrap ? (e.state.gzhead = t,
            z) : P
        }
        ,
        t.deflate = function(e, t) {
            var r, s, u, h;
            if (!e || !e.state || t > D || 0 > t)
                return e ? n(e, P) : P;
            if (s = e.state,
            !e.output || !e.input && 0 !== e.avail_in || s.status === oe && t !== R)
                return n(e, 0 === e.avail_out ? O : P);
            if (s.strm = e,
            r = s.last_flush,
            s.last_flush = t,
            s.status === ee)
                if (2 === s.wrap)
                    e.adler = 0,
                    l(s, 31),
                    l(s, 139),
                    l(s, 8),
                    s.gzhead ? (l(s, (s.gzhead.text ? 1 : 0) + (s.gzhead.hcrc ? 2 : 0) + (s.gzhead.extra ? 4 : 0) + (s.gzhead.name ? 8 : 0) + (s.gzhead.comment ? 16 : 0)),
                    l(s, 255 & s.gzhead.time),
                    l(s, 255 & s.gzhead.time >> 8),
                    l(s, 255 & s.gzhead.time >> 16),
                    l(s, 255 & s.gzhead.time >> 24),
                    l(s, 9 === s.level ? 2 : s.strategy >= U || 2 > s.level ? 4 : 0),
                    l(s, 255 & s.gzhead.os),
                    s.gzhead.extra && s.gzhead.extra.length && (l(s, 255 & s.gzhead.extra.length),
                    l(s, 255 & s.gzhead.extra.length >> 8)),
                    s.gzhead.hcrc && (e.adler = S(e.adler, s.pending_buf, s.pending, 0)),
                    s.gzindex = 0,
                    s.status = te) : (l(s, 0),
                    l(s, 0),
                    l(s, 0),
                    l(s, 0),
                    l(s, 0),
                    l(s, 9 === s.level ? 2 : s.strategy >= U || 2 > s.level ? 4 : 0),
                    l(s, he),
                    s.status = ae);
                else {
                    var f = H + (s.w_bits - 8 << 4) << 8;
                    f |= (s.strategy >= U || 2 > s.level ? 0 : 6 > s.level ? 1 : 6 === s.level ? 2 : 3) << 6,
                    0 !== s.strstart && (f |= J),
                    f += 31 - f % 31,
                    s.status = ae,
                    c(s, f),
                    0 !== s.strstart && (c(s, e.adler >>> 16),
                    c(s, 65535 & e.adler)),
                    e.adler = 1
                }
            if (s.status === te)
                if (s.gzhead.extra) {
                    for (u = s.pending; s.gzindex < (65535 & s.gzhead.extra.length) && (s.pending !== s.pending_buf_size || (s.gzhead.hcrc && s.pending > u && (e.adler = S(e.adler, s.pending_buf, s.pending - u, u)),
                    o(e),
                    u = s.pending,
                    s.pending !== s.pending_buf_size)); )
                        l(s, 255 & s.gzhead.extra[s.gzindex]),
                        s.gzindex++;
                    s.gzhead.hcrc && s.pending > u && (e.adler = S(e.adler, s.pending_buf, s.pending - u, u)),
                    s.gzindex === s.gzhead.extra.length && (s.gzindex = 0,
                    s.status = re)
                } else
                    s.status = re;
            if (s.status === re)
                if (s.gzhead.name) {
                    u = s.pending;
                    do {
                        if (s.pending === s.pending_buf_size && (s.gzhead.hcrc && s.pending > u && (e.adler = S(e.adler, s.pending_buf, s.pending - u, u)),
                        o(e),
                        u = s.pending,
                        s.pending === s.pending_buf_size)) {
                            h = 1;
                            break
                        }
                        h = s.gzindex < s.gzhead.name.length ? 255 & s.gzhead.name.charCodeAt(s.gzindex++) : 0,
                        l(s, h)
                    } while (0 !== h);
                    s.gzhead.hcrc && s.pending > u && (e.adler = S(e.adler, s.pending_buf, s.pending - u, u)),
                    0 === h && (s.gzindex = 0,
                    s.status = ne)
                } else
                    s.status = ne;
            if (s.status === ne)
                if (s.gzhead.comment) {
                    u = s.pending;
                    do {
                        if (s.pending === s.pending_buf_size && (s.gzhead.hcrc && s.pending > u && (e.adler = S(e.adler, s.pending_buf, s.pending - u, u)),
                        o(e),
                        u = s.pending,
                        s.pending === s.pending_buf_size)) {
                            h = 1;
                            break
                        }
                        h = s.gzindex < s.gzhead.comment.length ? 255 & s.gzhead.comment.charCodeAt(s.gzindex++) : 0,
                        l(s, h)
                    } while (0 !== h);
                    s.gzhead.hcrc && s.pending > u && (e.adler = S(e.adler, s.pending_buf, s.pending - u, u)),
                    0 === h && (s.status = ie)
                } else
                    s.status = ie;
            if (s.status === ie && (s.gzhead.hcrc ? (s.pending + 2 > s.pending_buf_size && o(e),
            s.pending + 2 <= s.pending_buf_size && (l(s, 255 & e.adler),
            l(s, 255 & e.adler >> 8),
            e.adler = 0,
            s.status = ae)) : s.status = ae),
            0 !== s.pending) {
                if (o(e),
                0 === e.avail_out)
                    return s.last_flush = -1,
                    z
            } else if (0 === e.avail_in && i(t) <= i(r) && t !== R)
                return n(e, O);
            if (s.status === oe && 0 !== e.avail_in)
                return n(e, O);
            if (0 !== e.avail_in || 0 !== s.lookahead || t !== I && s.status !== oe) {
                var d = s.strategy === U ? m(s, t) : s.strategy === M ? _(s, t) : E[s.level].func(s, t);
                if ((d === ce || d === ue) && (s.status = oe),
                d === se || d === ce)
                    return 0 === e.avail_out && (s.last_flush = -1),
                    z;
                if (d === le && (t === C ? x._tr_align(s) : t !== D && (x._tr_stored_block(s, 0, 0, !1),
                t === B && (a(s.head),
                0 === s.lookahead && (s.strstart = 0,
                s.block_start = 0,
                s.insert = 0))),
                o(e),
                0 === e.avail_out))
                    return s.last_flush = -1,
                    z
            }
            return t === R ? 0 >= s.wrap ? N : (2 === s.wrap ? (l(s, 255 & e.adler),
            l(s, 255 & e.adler >> 8),
            l(s, 255 & e.adler >> 16),
            l(s, 255 & e.adler >> 24),
            l(s, 255 & e.total_in),
            l(s, 255 & e.total_in >> 8),
            l(s, 255 & e.total_in >> 16),
            l(s, 255 & e.total_in >> 24)) : (c(s, e.adler >>> 16),
            c(s, 65535 & e.adler)),
            o(e),
            0 < s.wrap && (s.wrap = -s.wrap),
            0 === s.pending ? N : z) : z
        }
        ,
        t.deflateEnd = function(e) {
            var t;
            return e && e.state ? (t = e.state.status) !== ee && t !== te && t !== re && t !== ne && t !== ie && t !== ae && t !== oe ? n(e, P) : (e.state = null,
            t === ae ? n(e, -3) : z) : P
        }
        ,
        t.deflateSetDictionary = function(e, t) {
            var r, n, i, o, s, l, c, u, h = t.length;
            if (!e || !e.state)
                return P;
            if (2 === (o = (r = e.state).wrap) || 1 === o && r.status !== ee || r.lookahead)
                return P;
            for (1 === o && (e.adler = A(e.adler, t, h, 0)),
            r.wrap = 0,
            h >= r.w_size && (0 === o && (a(r.head),
            r.strstart = 0,
            r.block_start = 0,
            r.insert = 0),
            u = new k.Buf8(r.w_size),
            k.arraySet(u, t, h - r.w_size, r.w_size, 0),
            t = u,
            h = r.w_size),
            s = e.avail_in,
            l = e.next_in,
            c = e.input,
            e.avail_in = h,
            e.next_in = 0,
            e.input = t,
            f(r); r.lookahead >= K; ) {
                n = r.strstart,
                i = r.lookahead - (K - 1);
                do {
                    r.ins_h = (r.ins_h << r.hash_shift ^ r.window[n + K - 1]) & r.hash_mask,
                    r.prev[n & r.w_mask] = r.head[r.ins_h],
                    r.head[r.ins_h] = n,
                    n++
                } while (--i);
                r.strstart = n,
                r.lookahead = K - 1,
                f(r)
            }
            return r.strstart += r.lookahead,
            r.block_start = r.strstart,
            r.insert = r.lookahead,
            r.lookahead = 0,
            r.match_length = r.prev_length = K - 1,
            r.match_available = 0,
            e.next_in = l,
            e.input = c,
            e.avail_in = s,
            r.wrap = o,
            z
        }
        ,
        t.deflateInfo = "pako deflate (from Nodeca project)"
    }
    , function(e, t, r) {
        "use strict";
        function n(e) {
            for (var t = e.length; 0 <= --t; )
                e[t] = 0
        }
        function i(e, t, r, n, i) {
            this.static_tree = e,
            this.extra_bits = t,
            this.extra_base = r,
            this.elems = n,
            this.max_length = i,
            this.has_stree = e && e.length
        }
        function a(e, t) {
            this.dyn_tree = e,
            this.max_code = 0,
            this.stat_desc = t
        }
        function o(e) {
            return 256 > e ? V[e] : V[256 + (e >>> 7)]
        }
        function s(e, t) {
            e.pending_buf[e.pending++] = 255 & t,
            e.pending_buf[e.pending++] = 255 & t >>> 8
        }
        function l(e, t, r) {
            e.bi_valid > D - r ? (e.bi_buf |= 65535 & t << e.bi_valid,
            s(e, e.bi_buf),
            e.bi_buf = t >> D - e.bi_valid,
            e.bi_valid += r - D) : (e.bi_buf |= 65535 & t << e.bi_valid,
            e.bi_valid += r)
        }
        function c(e, t, r) {
            l(e, r[2 * t], r[2 * t + 1])
        }
        function u(e, t) {
            var r = 0;
            do {
                r |= 1 & e,
                e >>>= 1,
                r <<= 1
            } while (0 < --t);
            return r >>> 1
        }
        function h(e, t, r) {
            var n, i, a = Array(R + 1), o = 0;
            for (n = 1; n <= R; n++)
                a[n] = o = o + r[n - 1] << 1;
            for (i = 0; i <= t; i++) {
                var s = e[2 * i + 1];
                0 !== s && (e[2 * i] = u(a[s]++, s))
            }
        }
        function f(e) {
            var t;
            for (t = 0; t < T; t++)
                e.dyn_ltree[2 * t] = 0;
            for (t = 0; t < I; t++)
                e.dyn_dtree[2 * t] = 0;
            for (t = 0; t < C; t++)
                e.bl_tree[2 * t] = 0;
            e.dyn_ltree[2 * N] = 1,
            e.opt_len = e.static_len = 0,
            e.last_lit = e.matches = 0
        }
        function d(e) {
            8 < e.bi_valid ? s(e, e.bi_buf) : 0 < e.bi_valid && (e.pending_buf[e.pending++] = e.bi_buf),
            e.bi_buf = 0,
            e.bi_valid = 0
        }
        function p(e, t, r, n) {
            var i = 2 * t
              , a = 2 * r;
            return e[i] < e[a] || e[i] === e[a] && n[t] <= n[r]
        }
        function _(e, t, r) {
            for (var n = e.heap[r], i = r << 1; i <= e.heap_len && (i < e.heap_len && p(t, e.heap[i + 1], e.heap[i], e.depth) && i++,
            !p(t, n, e.heap[i], e.depth)); )
                e.heap[r] = e.heap[i],
                r = i,
                i <<= 1;
            e.heap[r] = n
        }
        function m(e, t, r) {
            var n, i, a, s, u = 0;
            if (0 !== e.last_lit)
                do {
                    n = e.pending_buf[e.d_buf + 2 * u] << 8 | e.pending_buf[e.d_buf + 2 * u + 1],
                    i = e.pending_buf[e.l_buf + u],
                    u++,
                    0 === n ? c(e, i, t) : (c(e, (a = $[i]) + S + 1, t),
                    0 !== (s = L[a]) && l(e, i -= G[a], s),
                    c(e, a = o(--n), r),
                    0 !== (s = U[a]) && l(e, n -= X[a], s))
                } while (u < e.last_lit);
            c(e, N, t)
        }
        function w(e, t) {
            var r, n, i, a = t.dyn_tree, o = t.stat_desc.static_tree, s = t.stat_desc.has_stree, l = t.stat_desc.elems, c = -1;
            for (e.heap_len = 0,
            e.heap_max = B,
            r = 0; r < l; r++)
                0 === a[2 * r] ? a[2 * r + 1] = 0 : (e.heap[++e.heap_len] = c = r,
                e.depth[r] = 0);
            for (; 2 > e.heap_len; )
                a[2 * (i = e.heap[++e.heap_len] = 2 > c ? ++c : 0)] = 1,
                e.depth[i] = 0,
                e.opt_len--,
                s && (e.static_len -= o[2 * i + 1]);
            for (t.max_code = c,
            r = e.heap_len >> 1; 1 <= r; r--)
                _(e, a, r);
            i = l;
            do {
                r = e.heap[1],
                e.heap[1] = e.heap[e.heap_len--],
                _(e, a, 1),
                n = e.heap[1],
                e.heap[--e.heap_max] = r,
                e.heap[--e.heap_max] = n,
                a[2 * i] = a[2 * r] + a[2 * n],
                e.depth[i] = (e.depth[r] >= e.depth[n] ? e.depth[r] : e.depth[n]) + 1,
                a[2 * r + 1] = a[2 * n + 1] = i,
                e.heap[1] = i++,
                _(e, a, 1)
            } while (2 <= e.heap_len);
            e.heap[--e.heap_max] = e.heap[1],
            function(e, t) {
                var r, n, i, a, o, s, l = t.dyn_tree, c = t.max_code, u = t.stat_desc.static_tree, h = t.stat_desc.has_stree, f = t.stat_desc.extra_bits, d = t.stat_desc.extra_base, p = t.stat_desc.max_length, _ = 0;
                for (a = 0; a <= R; a++)
                    e.bl_count[a] = 0;
                for (l[2 * e.heap[e.heap_max] + 1] = 0,
                r = e.heap_max + 1; r < B; r++)
                    (a = l[2 * l[2 * (n = e.heap[r]) + 1] + 1] + 1) > p && (a = p,
                    _++),
                    l[2 * n + 1] = a,
                    !(n > c) && (e.bl_count[a]++,
                    o = 0,
                    n >= d && (o = f[n - d]),
                    s = l[2 * n],
                    e.opt_len += s * (a + o),
                    h && (e.static_len += s * (u[2 * n + 1] + o)));
                if (0 != _) {
                    do {
                        for (a = p - 1; 0 === e.bl_count[a]; )
                            a--;
                        e.bl_count[a]--,
                        e.bl_count[a + 1] += 2,
                        e.bl_count[p]--,
                        _ -= 2
                    } while (0 < _);
                    for (a = p; 0 !== a; a--)
                        for (n = e.bl_count[a]; 0 !== n; )
                            !((i = e.heap[--r]) > c) && (l[2 * i + 1] !== a && (e.opt_len += (a - l[2 * i + 1]) * l[2 * i],
                            l[2 * i + 1] = a),
                            n--)
                }
            }(e, t),
            h(a, c, e.bl_count)
        }
        function b(e, t, r) {
            var n, i, a = -1, o = t[1], s = 0, l = 7, c = 4;
            for (0 === o && (l = 138,
            c = 3),
            t[2 * (r + 1) + 1] = 65535,
            n = 0; n <= r; n++)
                i = o,
                o = t[2 * (n + 1) + 1],
                ++s < l && i === o || (s < c ? e.bl_tree[2 * i] += s : 0 === i ? 10 >= s ? e.bl_tree[2 * O]++ : e.bl_tree[2 * F]++ : (i !== a && e.bl_tree[2 * i]++,
                e.bl_tree[2 * P]++),
                s = 0,
                a = i,
                0 === o ? (l = 138,
                c = 3) : i === o ? (l = 6,
                c = 3) : (l = 7,
                c = 4))
        }
        function g(e, t, r) {
            var n, i, a = -1, o = t[1], s = 0, u = 7, h = 4;
            for (0 === o && (u = 138,
            h = 3),
            n = 0; n <= r; n++)
                if (i = o,
                o = t[2 * (n + 1) + 1],
                !(++s < u && i === o)) {
                    if (s < h)
                        do {
                            c(e, i, e.bl_tree)
                        } while (0 != --s);
                    else
                        0 === i ? 10 >= s ? (c(e, O, e.bl_tree),
                        l(e, s - 3, 3)) : (c(e, F, e.bl_tree),
                        l(e, s - 11, 7)) : (i !== a && (c(e, i, e.bl_tree),
                        s--),
                        c(e, P, e.bl_tree),
                        l(e, s - 3, 2));
                    s = 0,
                    a = i,
                    0 === o ? (u = 138,
                    h = 3) : i === o ? (u = 6,
                    h = 3) : (u = 7,
                    h = 4)
                }
        }
        function y(e, t, r, n) {
            l(e, (x << 1) + (n ? 1 : 0), 3),
            function(e, t, r, n) {
                d(e),
                n && (s(e, r),
                s(e, ~r)),
                v.arraySet(e.pending_buf, e.window, t, r, e.pending),
                e.pending += r
            }(e, t, r, !0)
        }
        var v = r(1)
          , E = 0
          , k = 1
          , x = 0
          , A = 29
          , S = 256
          , T = S + 1 + A
          , I = 30
          , C = 19
          , B = 2 * T + 1
          , R = 15
          , D = 16
          , z = 7
          , N = 256
          , P = 16
          , O = 17
          , F = 18
          , L = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
          , U = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
          , M = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7]
          , j = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
          , W = Array(2 * (T + 2));
        n(W);
        var H = Array(2 * I);
        n(H);
        var V = Array(512);
        n(V);
        var $ = Array(256);
        n($);
        var G = Array(A);
        n(G);
        var X = Array(I);
        n(X);
        var Y, Z, K, q = !1;
        t._tr_init = function(e) {
            q || (function() {
                var e, t, r, n, a, o = Array(R + 1);
                for (r = 0,
                n = 0; n < A - 1; n++)
                    for (G[n] = r,
                    e = 0; e < 1 << L[n]; e++)
                        $[r++] = n;
                for ($[r - 1] = n,
                a = 0,
                n = 0; 16 > n; n++)
                    for (X[n] = a,
                    e = 0; e < 1 << U[n]; e++)
                        V[a++] = n;
                for (a >>= 7; n < I; n++)
                    for (X[n] = a << 7,
                    e = 0; e < 1 << U[n] - 7; e++)
                        V[256 + a++] = n;
                for (t = 0; t <= R; t++)
                    o[t] = 0;
                for (e = 0; 143 >= e; )
                    W[2 * e + 1] = 8,
                    e++,
                    o[8]++;
                for (; 255 >= e; )
                    W[2 * e + 1] = 9,
                    e++,
                    o[9]++;
                for (; 279 >= e; )
                    W[2 * e + 1] = 7,
                    e++,
                    o[7]++;
                for (; 287 >= e; )
                    W[2 * e + 1] = 8,
                    e++,
                    o[8]++;
                for (h(W, T + 1, o),
                e = 0; e < I; e++)
                    H[2 * e + 1] = 5,
                    H[2 * e] = u(e, 5);
                Y = new i(W,L,S + 1,T,R),
                Z = new i(H,U,0,I,R),
                K = new i([],M,0,C,z)
            }(),
            q = !0),
            e.l_desc = new a(e.dyn_ltree,Y),
            e.d_desc = new a(e.dyn_dtree,Z),
            e.bl_desc = new a(e.bl_tree,K),
            e.bi_buf = 0,
            e.bi_valid = 0,
            f(e)
        }
        ,
        t._tr_stored_block = y,
        t._tr_flush_block = function(e, t, r, n) {
            var i, a, o = 0;
            0 < e.level ? (2 === e.strm.data_type && (e.strm.data_type = function(e) {
                var t, r = 4093624447;
                for (t = 0; 31 >= t; t++,
                r >>>= 1)
                    if (1 & r && 0 !== e.dyn_ltree[2 * t])
                        return E;
                if (0 !== e.dyn_ltree[18] || 0 !== e.dyn_ltree[20] || 0 !== e.dyn_ltree[26])
                    return k;
                for (t = 32; t < S; t++)
                    if (0 !== e.dyn_ltree[2 * t])
                        return k;
                return E
            }(e)),
            w(e, e.l_desc),
            w(e, e.d_desc),
            o = function(e) {
                var t;
                for (b(e, e.dyn_ltree, e.l_desc.max_code),
                b(e, e.dyn_dtree, e.d_desc.max_code),
                w(e, e.bl_desc),
                t = C - 1; 3 <= t && 0 === e.bl_tree[2 * j[t] + 1]; t--)
                    ;
                return e.opt_len += 3 * (t + 1) + 5 + 5 + 4,
                t
            }(e),
            i = e.opt_len + 3 + 7 >>> 3,
            (a = e.static_len + 3 + 7 >>> 3) <= i && (i = a)) : i = a = r + 5,
            r + 4 <= i && -1 !== t ? y(e, t, r, n) : 4 === e.strategy || a === i ? (l(e, 2 + (n ? 1 : 0), 3),
            m(e, W, H)) : (l(e, 4 + (n ? 1 : 0), 3),
            function(e, t, r, n) {
                var i;
                for (l(e, t - 257, 5),
                l(e, r - 1, 5),
                l(e, n - 4, 4),
                i = 0; i < n; i++)
                    l(e, e.bl_tree[2 * j[i] + 1], 3);
                g(e, e.dyn_ltree, t - 1),
                g(e, e.dyn_dtree, r - 1)
            }(e, e.l_desc.max_code + 1, e.d_desc.max_code + 1, o + 1),
            m(e, e.dyn_ltree, e.dyn_dtree)),
            f(e),
            n && d(e)
        }
        ,
        t._tr_tally = function(e, t, r) {
            return e.pending_buf[e.d_buf + 2 * e.last_lit] = 255 & t >>> 8,
            e.pending_buf[e.d_buf + 2 * e.last_lit + 1] = 255 & t,
            e.pending_buf[e.l_buf + e.last_lit] = 255 & r,
            e.last_lit++,
            0 === t ? e.dyn_ltree[2 * r]++ : (e.matches++,
            t--,
            e.dyn_ltree[2 * ($[r] + S + 1)]++,
            e.dyn_dtree[2 * o(t)]++),
            e.last_lit === e.lit_bufsize - 1
        }
        ,
        t._tr_align = function(e) {
            l(e, 2, 3),
            c(e, N, W),
            function(e) {
                16 === e.bi_valid ? (s(e, e.bi_buf),
                e.bi_buf = 0,
                e.bi_valid = 0) : 8 <= e.bi_valid && (e.pending_buf[e.pending++] = 255 & e.bi_buf,
                e.bi_buf >>= 8,
                e.bi_valid -= 8)
            }(e)
        }
    }
    , function(e, t, r) {
        "use strict";
        function n(e) {
            if (!(this instanceof n))
                return new n(e);
            this.options = o.assign({
                chunkSize: 16384,
                windowBits: 0,
                to: ""
            }, e || {});
            var t = this.options;
            t.raw && 0 <= t.windowBits && 16 > t.windowBits && (t.windowBits = -t.windowBits,
            0 === t.windowBits && (t.windowBits = -15)),
            0 <= t.windowBits && 16 > t.windowBits && !(e && e.windowBits) && (t.windowBits += 32),
            15 < t.windowBits && 48 > t.windowBits && 0 == (15 & t.windowBits) && (t.windowBits |= 15),
            this.err = 0,
            this.msg = "",
            this.ended = !1,
            this.chunks = [],
            this.strm = new u,
            this.strm.avail_out = 0;
            var r = a.inflateInit2(this.strm, t.windowBits);
            if (r !== l.Z_OK)
                throw new Error(c[r]);
            if (this.header = new h,
            a.inflateGetHeader(this.strm, this.header),
            t.dictionary && ("string" == typeof t.dictionary ? t.dictionary = s.string2buf(t.dictionary) : "[object ArrayBuffer]" === f.call(t.dictionary) && (t.dictionary = new Uint8Array(t.dictionary)),
            t.raw && (r = a.inflateSetDictionary(this.strm, t.dictionary)) !== l.Z_OK))
                throw new Error(c[r])
        }
        function i(e, t) {
            var r = new n(t);
            if (r.push(e, !0),
            r.err)
                throw r.msg || c[r.err];
            return r.result
        }
        var a = r(15)
          , o = r(1)
          , s = r(6)
          , l = r(8)
          , c = r(3)
          , u = r(7)
          , h = r(18)
          , f = Object.prototype.toString;
        n.prototype.push = function(e, t) {
            var r, n, i, c, u, h = this.strm, d = this.options.chunkSize, p = this.options.dictionary, _ = !1;
            if (this.ended)
                return !1;
            n = t === ~~t ? t : !0 === t ? l.Z_FINISH : l.Z_NO_FLUSH,
            h.input = "string" == typeof e ? s.binstring2buf(e) : "[object ArrayBuffer]" === f.call(e) ? new Uint8Array(e) : e,
            h.next_in = 0,
            h.avail_in = h.input.length;
            do {
                if (0 === h.avail_out && (h.output = new o.Buf8(d),
                h.next_out = 0,
                h.avail_out = d),
                (r = a.inflate(h, l.Z_NO_FLUSH)) === l.Z_NEED_DICT && p && (r = a.inflateSetDictionary(this.strm, p)),
                r === l.Z_BUF_ERROR && 1 == _ && (r = l.Z_OK,
                _ = !1),
                r !== l.Z_STREAM_END && r !== l.Z_OK)
                    return this.onEnd(r),
                    this.ended = !0,
                    !1;
                h.next_out && (0 === h.avail_out || r === l.Z_STREAM_END || 0 === h.avail_in && (n === l.Z_FINISH || n === l.Z_SYNC_FLUSH)) && ("string" === this.options.to ? (i = s.utf8border(h.output, h.next_out),
                c = h.next_out - i,
                u = s.buf2string(h.output, i),
                h.next_out = c,
                h.avail_out = d - c,
                c && o.arraySet(h.output, h.output, i, c, 0),
                this.onData(u)) : this.onData(o.shrinkBuf(h.output, h.next_out))),
                0 === h.avail_in && 0 === h.avail_out && (_ = !0)
            } while ((0 < h.avail_in || 0 === h.avail_out) && r !== l.Z_STREAM_END);
            return r === l.Z_STREAM_END && (n = l.Z_FINISH),
            n === l.Z_FINISH ? (r = a.inflateEnd(this.strm),
            this.onEnd(r),
            this.ended = !0,
            r === l.Z_OK) : n !== l.Z_SYNC_FLUSH || (this.onEnd(l.Z_OK),
            h.avail_out = 0,
            !0)
        }
        ,
        n.prototype.onData = function(e) {
            this.chunks.push(e)
        }
        ,
        n.prototype.onEnd = function(e) {
            e === l.Z_OK && ("string" === this.options.to ? this.result = this.chunks.join("") : this.result = o.flattenChunks(this.chunks)),
            this.chunks = [],
            this.err = e,
            this.msg = this.strm.msg
        }
        ,
        t.Inflate = n,
        t.inflate = i,
        t.inflateRaw = function(e, t) {
            return (t = t || {}).raw = !0,
            i(e, t)
        }
        ,
        t.ungzip = i
    }
    , function(e, t, r) {
        "use strict";
        function n(e) {
            return (255 & e >>> 24) + (65280 & e >>> 8) + ((65280 & e) << 8) + ((255 & e) << 24)
        }
        function i() {
            this.mode = 0,
            this.last = !1,
            this.wrap = 0,
            this.havedict = !1,
            this.flags = 0,
            this.dmax = 0,
            this.check = 0,
            this.total = 0,
            this.head = null,
            this.wbits = 0,
            this.wsize = 0,
            this.whave = 0,
            this.wnext = 0,
            this.window = null,
            this.hold = 0,
            this.bits = 0,
            this.length = 0,
            this.offset = 0,
            this.extra = 0,
            this.lencode = null,
            this.distcode = null,
            this.lenbits = 0,
            this.distbits = 0,
            this.ncode = 0,
            this.nlen = 0,
            this.ndist = 0,
            this.have = 0,
            this.next = null,
            this.lens = new d.Buf16(320),
            this.work = new d.Buf16(288),
            this.lendyn = null,
            this.distdyn = null,
            this.sane = 0,
            this.back = 0,
            this.was = 0
        }
        function a(e) {
            var t;
            return e && e.state ? (t = e.state,
            e.total_in = e.total_out = t.total = 0,
            e.msg = "",
            t.wrap && (e.adler = 1 & t.wrap),
            t.mode = E,
            t.last = 0,
            t.havedict = 0,
            t.dmax = 32768,
            t.head = null,
            t.hold = 0,
            t.bits = 0,
            t.lencode = t.lendyn = new d.Buf32(x),
            t.distcode = t.distdyn = new d.Buf32(A),
            t.sane = 1,
            t.back = -1,
            y) : v
        }
        function o(e) {
            var t;
            return e && e.state ? ((t = e.state).wsize = 0,
            t.whave = 0,
            t.wnext = 0,
            a(e)) : v
        }
        function s(e, t) {
            var r, n;
            return e && e.state ? (n = e.state,
            0 > t ? (r = 0,
            t = -t) : (r = 1 + (t >> 4),
            48 > t && (t &= 15)),
            t && (8 > t || 15 < t) ? v : (null !== n.window && n.wbits !== t && (n.window = null),
            n.wrap = r,
            n.wbits = t,
            o(e))) : v
        }
        function l(e, t) {
            var r, n;
            return e ? (n = new i,
            e.state = n,
            n.window = null,
            (r = s(e, t)) !== y && (e.state = null),
            r) : v
        }
        function c(e) {
            if (S) {
                var t;
                for (h = new d.Buf32(512),
                f = new d.Buf32(32),
                t = 0; 144 > t; )
                    e.lens[t++] = 8;
                for (; 256 > t; )
                    e.lens[t++] = 9;
                for (; 280 > t; )
                    e.lens[t++] = 7;
                for (; 288 > t; )
                    e.lens[t++] = 8;
                for (w(b, e.lens, 0, 288, h, 0, e.work, {
                    bits: 9
                }),
                t = 0; 32 > t; )
                    e.lens[t++] = 5;
                w(g, e.lens, 0, 32, f, 0, e.work, {
                    bits: 5
                }),
                S = !1
            }
            e.lencode = h,
            e.lenbits = 9,
            e.distcode = f,
            e.distbits = 5
        }
        function u(e, t, r, n) {
            var i, a = e.state;
            return null === a.window && (a.wsize = 1 << a.wbits,
            a.wnext = 0,
            a.whave = 0,
            a.window = new d.Buf8(a.wsize)),
            n >= a.wsize ? (d.arraySet(a.window, t, r - a.wsize, a.wsize, 0),
            a.wnext = 0,
            a.whave = a.wsize) : ((i = a.wsize - a.wnext) > n && (i = n),
            d.arraySet(a.window, t, r - n, i, a.wnext),
            (n -= i) ? (d.arraySet(a.window, t, r - n, n, 0),
            a.wnext = n,
            a.whave = a.wsize) : (a.wnext += i,
            a.wnext === a.wsize && (a.wnext = 0),
            a.whave < a.wsize && (a.whave += i))),
            0
        }
        var h, f, d = r(1), p = r(4), _ = r(5), m = r(16), w = r(17), b = 1, g = 2, y = 0, v = -2, E = 1, k = 12, x = 852, A = 592, S = !0;
        t.inflateReset = o,
        t.inflateReset2 = s,
        t.inflateResetKeep = a,
        t.inflateInit = function(e) {
            return l(e, 15)
        }
        ,
        t.inflateInit2 = l,
        t.inflate = function(e, t) {
            var r, i, a, o, s, l, h, f, x, A, S, T, I, C, B, R, D, z, N, P, O, F, L, U, M = 0, j = new d.Buf8(4), W = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
            if (!e || !e.state || !e.output || !e.input && 0 !== e.avail_in)
                return v;
            (r = e.state).mode === k && (r.mode = 13),
            s = e.next_out,
            a = e.output,
            h = e.avail_out,
            o = e.next_in,
            i = e.input,
            l = e.avail_in,
            f = r.hold,
            x = r.bits,
            A = l,
            S = h,
            F = y;
            e: for (; ; )
                switch (r.mode) {
                case E:
                    if (0 === r.wrap) {
                        r.mode = 13;
                        break
                    }
                    for (; 16 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    if (2 & r.wrap && 35615 === f) {
                        r.check = 0,
                        j[0] = 255 & f,
                        j[1] = 255 & f >>> 8,
                        r.check = _(r.check, j, 2, 0),
                        f = 0,
                        x = 0,
                        r.mode = 2;
                        break
                    }
                    if (r.flags = 0,
                    r.head && (r.head.done = !1),
                    !(1 & r.wrap) || (((255 & f) << 8) + (f >> 8)) % 31) {
                        e.msg = "incorrect header check",
                        r.mode = 30;
                        break
                    }
                    if (8 != (15 & f)) {
                        e.msg = "unknown compression method",
                        r.mode = 30;
                        break
                    }
                    if (x -= 4,
                    O = 8 + (15 & (f >>>= 4)),
                    0 === r.wbits)
                        r.wbits = O;
                    else if (O > r.wbits) {
                        e.msg = "invalid window size",
                        r.mode = 30;
                        break
                    }
                    r.dmax = 1 << O,
                    e.adler = r.check = 1,
                    r.mode = 512 & f ? 10 : k,
                    f = 0,
                    x = 0;
                    break;
                case 2:
                    for (; 16 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    if (r.flags = f,
                    8 != (255 & r.flags)) {
                        e.msg = "unknown compression method",
                        r.mode = 30;
                        break
                    }
                    if (57344 & r.flags) {
                        e.msg = "unknown header flags set",
                        r.mode = 30;
                        break
                    }
                    r.head && (r.head.text = 1 & f >> 8),
                    512 & r.flags && (j[0] = 255 & f,
                    j[1] = 255 & f >>> 8,
                    r.check = _(r.check, j, 2, 0)),
                    f = 0,
                    x = 0,
                    r.mode = 3;
                case 3:
                    for (; 32 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    r.head && (r.head.time = f),
                    512 & r.flags && (j[0] = 255 & f,
                    j[1] = 255 & f >>> 8,
                    j[2] = 255 & f >>> 16,
                    j[3] = 255 & f >>> 24,
                    r.check = _(r.check, j, 4, 0)),
                    f = 0,
                    x = 0,
                    r.mode = 4;
                case 4:
                    for (; 16 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    r.head && (r.head.xflags = 255 & f,
                    r.head.os = f >> 8),
                    512 & r.flags && (j[0] = 255 & f,
                    j[1] = 255 & f >>> 8,
                    r.check = _(r.check, j, 2, 0)),
                    f = 0,
                    x = 0,
                    r.mode = 5;
                case 5:
                    if (1024 & r.flags) {
                        for (; 16 > x; ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        r.length = f,
                        r.head && (r.head.extra_len = f),
                        512 & r.flags && (j[0] = 255 & f,
                        j[1] = 255 & f >>> 8,
                        r.check = _(r.check, j, 2, 0)),
                        f = 0,
                        x = 0
                    } else
                        r.head && (r.head.extra = null);
                    r.mode = 6;
                case 6:
                    if (1024 & r.flags && ((T = r.length) > l && (T = l),
                    T && (r.head && (O = r.head.extra_len - r.length,
                    !r.head.extra && (r.head.extra = Array(r.head.extra_len)),
                    d.arraySet(r.head.extra, i, o, T, O)),
                    512 & r.flags && (r.check = _(r.check, i, T, o)),
                    l -= T,
                    o += T,
                    r.length -= T),
                    r.length))
                        break e;
                    r.length = 0,
                    r.mode = 7;
                case 7:
                    if (2048 & r.flags) {
                        if (0 === l)
                            break e;
                        T = 0;
                        do {
                            O = i[o + T++],
                            r.head && O && 65536 > r.length && (r.head.name += _StringfromCharCode(O))
                        } while (O && T < l);
                        if (512 & r.flags && (r.check = _(r.check, i, T, o)),
                        l -= T,
                        o += T,
                        O)
                            break e
                    } else
                        r.head && (r.head.name = null);
                    r.length = 0,
                    r.mode = 8;
                case 8:
                    if (4096 & r.flags) {
                        if (0 === l)
                            break e;
                        T = 0;
                        do {
                            O = i[o + T++],
                            r.head && O && 65536 > r.length && (r.head.comment += _StringfromCharCode(O))
                        } while (O && T < l);
                        if (512 & r.flags && (r.check = _(r.check, i, T, o)),
                        l -= T,
                        o += T,
                        O)
                            break e
                    } else
                        r.head && (r.head.comment = null);
                    r.mode = 9;
                case 9:
                    if (512 & r.flags) {
                        for (; 16 > x; ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        if (f !== (65535 & r.check)) {
                            e.msg = "header crc mismatch",
                            r.mode = 30;
                            break
                        }
                        f = 0,
                        x = 0
                    }
                    r.head && (r.head.hcrc = 1 & r.flags >> 9,
                    r.head.done = !0),
                    e.adler = r.check = 0,
                    r.mode = k;
                    break;
                case 10:
                    for (; 32 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    e.adler = r.check = n(f),
                    f = 0,
                    x = 0,
                    r.mode = 11;
                case 11:
                    if (0 === r.havedict)
                        return e.next_out = s,
                        e.avail_out = h,
                        e.next_in = o,
                        e.avail_in = l,
                        r.hold = f,
                        r.bits = x,
                        2;
                    e.adler = r.check = 1,
                    r.mode = k;
                case k:
                    if (5 === t || 6 === t)
                        break e;
                case 13:
                    if (r.last) {
                        f >>>= 7 & x,
                        x -= 7 & x,
                        r.mode = 27;
                        break
                    }
                    for (; 3 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    switch (r.last = 1 & f,
                    x -= 1,
                    3 & (f >>>= 1)) {
                    case 0:
                        r.mode = 14;
                        break;
                    case 1:
                        if (c(r),
                        r.mode = 20,
                        6 === t) {
                            f >>>= 2,
                            x -= 2;
                            break e
                        }
                        break;
                    case 2:
                        r.mode = 17;
                        break;
                    case 3:
                        e.msg = "invalid block type",
                        r.mode = 30
                    }
                    f >>>= 2,
                    x -= 2;
                    break;
                case 14:
                    for (f >>>= 7 & x,
                    x -= 7 & x; 32 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    if ((65535 & f) != (65535 ^ f >>> 16)) {
                        e.msg = "invalid stored block lengths",
                        r.mode = 30;
                        break
                    }
                    if (r.length = 65535 & f,
                    f = 0,
                    x = 0,
                    r.mode = 15,
                    6 === t)
                        break e;
                case 15:
                    r.mode = 16;
                case 16:
                    if (T = r.length) {
                        if (T > l && (T = l),
                        T > h && (T = h),
                        0 === T)
                            break e;
                        d.arraySet(a, i, o, T, s),
                        l -= T,
                        o += T,
                        h -= T,
                        s += T,
                        r.length -= T;
                        break
                    }
                    r.mode = k;
                    break;
                case 17:
                    for (; 14 > x; ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    if (r.nlen = 257 + (31 & f),
                    f >>>= 5,
                    x -= 5,
                    r.ndist = 1 + (31 & f),
                    f >>>= 5,
                    x -= 5,
                    r.ncode = 4 + (15 & f),
                    f >>>= 4,
                    x -= 4,
                    286 < r.nlen || 30 < r.ndist) {
                        e.msg = "too many length or distance symbols",
                        r.mode = 30;
                        break
                    }
                    r.have = 0,
                    r.mode = 18;
                case 18:
                    for (; r.have < r.ncode; ) {
                        for (; 3 > x; ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        r.lens[W[r.have++]] = 7 & f,
                        f >>>= 3,
                        x -= 3
                    }
                    for (; 19 > r.have; )
                        r.lens[W[r.have++]] = 0;
                    if (r.lencode = r.lendyn,
                    r.lenbits = 7,
                    L = {
                        bits: r.lenbits
                    },
                    F = w(0, r.lens, 0, 19, r.lencode, 0, r.work, L),
                    r.lenbits = L.bits,
                    F) {
                        e.msg = "invalid code lengths set",
                        r.mode = 30;
                        break
                    }
                    r.have = 0,
                    r.mode = 19;
                case 19:
                    for (; r.have < r.nlen + r.ndist; ) {
                        for (; R = 255 & (M = r.lencode[f & (1 << r.lenbits) - 1]) >>> 16,
                        D = 65535 & M,
                        !((B = M >>> 24) <= x); ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        if (16 > D)
                            f >>>= B,
                            x -= B,
                            r.lens[r.have++] = D;
                        else {
                            if (16 === D) {
                                for (U = B + 2; x < U; ) {
                                    if (0 === l)
                                        break e;
                                    l--,
                                    f += i[o++] << x,
                                    x += 8
                                }
                                if (f >>>= B,
                                x -= B,
                                0 === r.have) {
                                    e.msg = "invalid bit length repeat",
                                    r.mode = 30;
                                    break
                                }
                                O = r.lens[r.have - 1],
                                T = 3 + (3 & f),
                                f >>>= 2,
                                x -= 2
                            } else if (17 === D) {
                                for (U = B + 3; x < U; ) {
                                    if (0 === l)
                                        break e;
                                    l--,
                                    f += i[o++] << x,
                                    x += 8
                                }
                                x -= B,
                                O = 0,
                                T = 3 + (7 & (f >>>= B)),
                                f >>>= 3,
                                x -= 3
                            } else {
                                for (U = B + 7; x < U; ) {
                                    if (0 === l)
                                        break e;
                                    l--,
                                    f += i[o++] << x,
                                    x += 8
                                }
                                x -= B,
                                O = 0,
                                T = 11 + (127 & (f >>>= B)),
                                f >>>= 7,
                                x -= 7
                            }
                            if (r.have + T > r.nlen + r.ndist) {
                                e.msg = "invalid bit length repeat",
                                r.mode = 30;
                                break
                            }
                            for (; T--; )
                                r.lens[r.have++] = O
                        }
                    }
                    if (30 === r.mode)
                        break;
                    if (0 === r.lens[256]) {
                        e.msg = "invalid code -- missing end-of-block",
                        r.mode = 30;
                        break
                    }
                    if (r.lenbits = 9,
                    L = {
                        bits: r.lenbits
                    },
                    F = w(b, r.lens, 0, r.nlen, r.lencode, 0, r.work, L),
                    r.lenbits = L.bits,
                    F) {
                        e.msg = "invalid literal/lengths set",
                        r.mode = 30;
                        break
                    }
                    if (r.distbits = 6,
                    r.distcode = r.distdyn,
                    L = {
                        bits: r.distbits
                    },
                    F = w(g, r.lens, r.nlen, r.ndist, r.distcode, 0, r.work, L),
                    r.distbits = L.bits,
                    F) {
                        e.msg = "invalid distances set",
                        r.mode = 30;
                        break
                    }
                    if (r.mode = 20,
                    6 === t)
                        break e;
                case 20:
                    r.mode = 21;
                case 21:
                    if (6 <= l && 258 <= h) {
                        e.next_out = s,
                        e.avail_out = h,
                        e.next_in = o,
                        e.avail_in = l,
                        r.hold = f,
                        r.bits = x,
                        m(e, S),
                        s = e.next_out,
                        a = e.output,
                        h = e.avail_out,
                        o = e.next_in,
                        i = e.input,
                        l = e.avail_in,
                        f = r.hold,
                        x = r.bits,
                        r.mode === k && (r.back = -1);
                        break
                    }
                    for (r.back = 0; R = 255 & (M = r.lencode[f & (1 << r.lenbits) - 1]) >>> 16,
                    D = 65535 & M,
                    !((B = M >>> 24) <= x); ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    if (R && 0 == (240 & R)) {
                        for (z = B,
                        N = R,
                        P = D; R = 255 & (M = r.lencode[P + ((f & (1 << z + N) - 1) >> z)]) >>> 16,
                        D = 65535 & M,
                        !(z + (B = M >>> 24) <= x); ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        f >>>= z,
                        x -= z,
                        r.back += z
                    }
                    if (f >>>= B,
                    x -= B,
                    r.back += B,
                    r.length = D,
                    0 === R) {
                        r.mode = 26;
                        break
                    }
                    if (32 & R) {
                        r.back = -1,
                        r.mode = k;
                        break
                    }
                    if (64 & R) {
                        e.msg = "invalid literal/length code",
                        r.mode = 30;
                        break
                    }
                    r.extra = 15 & R,
                    r.mode = 22;
                case 22:
                    if (r.extra) {
                        for (U = r.extra; x < U; ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        r.length += f & (1 << r.extra) - 1,
                        f >>>= r.extra,
                        x -= r.extra,
                        r.back += r.extra
                    }
                    r.was = r.length,
                    r.mode = 23;
                case 23:
                    for (; R = 255 & (M = r.distcode[f & (1 << r.distbits) - 1]) >>> 16,
                    D = 65535 & M,
                    !((B = M >>> 24) <= x); ) {
                        if (0 === l)
                            break e;
                        l--,
                        f += i[o++] << x,
                        x += 8
                    }
                    if (0 == (240 & R)) {
                        for (z = B,
                        N = R,
                        P = D; R = 255 & (M = r.distcode[P + ((f & (1 << z + N) - 1) >> z)]) >>> 16,
                        D = 65535 & M,
                        !(z + (B = M >>> 24) <= x); ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        f >>>= z,
                        x -= z,
                        r.back += z
                    }
                    if (f >>>= B,
                    x -= B,
                    r.back += B,
                    64 & R) {
                        e.msg = "invalid distance code",
                        r.mode = 30;
                        break
                    }
                    r.offset = D,
                    r.extra = 15 & R,
                    r.mode = 24;
                case 24:
                    if (r.extra) {
                        for (U = r.extra; x < U; ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        r.offset += f & (1 << r.extra) - 1,
                        f >>>= r.extra,
                        x -= r.extra,
                        r.back += r.extra
                    }
                    if (r.offset > r.dmax) {
                        e.msg = "invalid distance too far back",
                        r.mode = 30;
                        break
                    }
                    r.mode = 25;
                case 25:
                    if (0 === h)
                        break e;
                    if (T = S - h,
                    r.offset > T) {
                        if ((T = r.offset - T) > r.whave && r.sane) {
                            e.msg = "invalid distance too far back",
                            r.mode = 30;
                            break
                        }
                        T > r.wnext ? (T -= r.wnext,
                        I = r.wsize - T) : I = r.wnext - T,
                        T > r.length && (T = r.length),
                        C = r.window
                    } else
                        C = a,
                        I = s - r.offset,
                        T = r.length;
                    T > h && (T = h),
                    h -= T,
                    r.length -= T;
                    do {
                        a[s++] = C[I++]
                    } while (--T);
                    0 === r.length && (r.mode = 21);
                    break;
                case 26:
                    if (0 === h)
                        break e;
                    a[s++] = r.length,
                    h--,
                    r.mode = 21;
                    break;
                case 27:
                    if (r.wrap) {
                        for (; 32 > x; ) {
                            if (0 === l)
                                break e;
                            l--,
                            f |= i[o++] << x,
                            x += 8
                        }
                        if (S -= h,
                        e.total_out += S,
                        r.total += S,
                        S && (e.adler = r.check = r.flags ? _(r.check, a, S, s - S) : p(r.check, a, S, s - S)),
                        S = h,
                        (r.flags ? f : n(f)) !== r.check) {
                            e.msg = "incorrect data check",
                            r.mode = 30;
                            break
                        }
                        f = 0,
                        x = 0
                    }
                    r.mode = 28;
                case 28:
                    if (r.wrap && r.flags) {
                        for (; 32 > x; ) {
                            if (0 === l)
                                break e;
                            l--,
                            f += i[o++] << x,
                            x += 8
                        }
                        if (f !== (4294967295 & r.total)) {
                            e.msg = "incorrect length check",
                            r.mode = 30;
                            break
                        }
                        f = 0,
                        x = 0
                    }
                    r.mode = 29;
                case 29:
                    F = 1;
                    break e;
                case 30:
                    F = -3;
                    break e;
                case 31:
                    return -4;
                case 32:
                default:
                    return v
                }
            return e.next_out = s,
            e.avail_out = h,
            e.next_in = o,
            e.avail_in = l,
            r.hold = f,
            r.bits = x,
            (r.wsize || S !== e.avail_out && r.mode < 30 && (r.mode < 27 || 4 !== t)) && u(e, e.output, e.next_out, S - e.avail_out) ? (r.mode = 31,
            -4) : (A -= e.avail_in,
            S -= e.avail_out,
            e.total_in += A,
            e.total_out += S,
            r.total += S,
            r.wrap && S && (e.adler = r.check = r.flags ? _(r.check, a, S, e.next_out - S) : p(r.check, a, S, e.next_out - S)),
            e.data_type = r.bits + (r.last ? 64 : 0) + (r.mode === k ? 128 : 0) + (20 === r.mode || 15 === r.mode ? 256 : 0),
            (0 === A && 0 === S || 4 === t) && F === y && (F = -5),
            F)
        }
        ,
        t.inflateEnd = function(e) {
            if (!e || !e.state)
                return v;
            var t = e.state;
            return t.window && (t.window = null),
            e.state = null,
            y
        }
        ,
        t.inflateGetHeader = function(e, t) {
            var r;
            return e && e.state ? 0 == (2 & (r = e.state).wrap) ? v : (r.head = t,
            t.done = !1,
            y) : v
        }
        ,
        t.inflateSetDictionary = function(e, t) {
            var r, n = t.length;
            return e && e.state ? 0 !== (r = e.state).wrap && 11 !== r.mode ? v : 11 === r.mode && p(1, t, n, 0) !== r.check ? -3 : u(e, t, n, n) ? (r.mode = 31,
            -4) : (r.havedict = 1,
            y) : v
        }
        ,
        t.inflateInfo = "pako inflate (from Nodeca project)"
    }
    , function(e) {
        "use strict";
        e.exports = function(e, t) {
            var r, n, i, a, o, s, l, c, u, h, f, d, p, _, m, w, b, g, y, v, E, k, x, A, S;
            r = e.state,
            n = e.next_in,
            A = e.input,
            i = n + (e.avail_in - 5),
            a = e.next_out,
            S = e.output,
            o = a - (t - e.avail_out),
            s = a + (e.avail_out - 257),
            l = r.dmax,
            c = r.wsize,
            u = r.whave,
            h = r.wnext,
            f = r.window,
            d = r.hold,
            p = r.bits,
            _ = r.lencode,
            m = r.distcode,
            w = (1 << r.lenbits) - 1,
            b = (1 << r.distbits) - 1;
            e: do {
                15 > p && (d += A[n++] << p,
                p += 8,
                d += A[n++] << p,
                p += 8),
                g = _[d & w];
                t: for (; ; ) {
                    if (d >>>= y = g >>> 24,
                    p -= y,
                    0 === (y = 255 & g >>> 16))
                        S[a++] = 65535 & g;
                    else {
                        if (!(16 & y)) {
                            if (0 == (64 & y)) {
                                g = _[(65535 & g) + (d & (1 << y) - 1)];
                                continue t
                            }
                            if (32 & y) {
                                r.mode = 12;
                                break e
                            }
                            e.msg = "invalid literal/length code",
                            r.mode = 30;
                            break e
                        }
                        v = 65535 & g,
                        (y &= 15) && (p < y && (d += A[n++] << p,
                        p += 8),
                        v += d & (1 << y) - 1,
                        d >>>= y,
                        p -= y),
                        15 > p && (d += A[n++] << p,
                        p += 8,
                        d += A[n++] << p,
                        p += 8),
                        g = m[d & b];
                        r: for (; ; ) {
                            if (d >>>= y = g >>> 24,
                            p -= y,
                            !(16 & (y = 255 & g >>> 16))) {
                                if (0 == (64 & y)) {
                                    g = m[(65535 & g) + (d & (1 << y) - 1)];
                                    continue r
                                }
                                e.msg = "invalid distance code",
                                r.mode = 30;
                                break e
                            }
                            if (E = 65535 & g,
                            p < (y &= 15) && (d += A[n++] << p,
                            (p += 8) < y && (d += A[n++] << p,
                            p += 8)),
                            (E += d & (1 << y) - 1) > l) {
                                e.msg = "invalid distance too far back",
                                r.mode = 30;
                                break e
                            }
                            if (d >>>= y,
                            p -= y,
                            E > (y = a - o)) {
                                if ((y = E - y) > u && r.sane) {
                                    e.msg = "invalid distance too far back",
                                    r.mode = 30;
                                    break e
                                }
                                if (k = 0,
                                x = f,
                                0 === h) {
                                    if (k += c - y,
                                    y < v) {
                                        v -= y;
                                        do {
                                            S[a++] = f[k++]
                                        } while (--y);
                                        k = a - E,
                                        x = S
                                    }
                                } else if (h < y) {
                                    if (k += c + h - y,
                                    (y -= h) < v) {
                                        v -= y;
                                        do {
                                            S[a++] = f[k++]
                                        } while (--y);
                                        if (k = 0,
                                        h < v) {
                                            v -= y = h;
                                            do {
                                                S[a++] = f[k++]
                                            } while (--y);
                                            k = a - E,
                                            x = S
                                        }
                                    }
                                } else if (k += h - y,
                                y < v) {
                                    v -= y;
                                    do {
                                        S[a++] = f[k++]
                                    } while (--y);
                                    k = a - E,
                                    x = S
                                }
                                for (; 2 < v; )
                                    S[a++] = x[k++],
                                    S[a++] = x[k++],
                                    S[a++] = x[k++],
                                    v -= 3;
                                v && (S[a++] = x[k++],
                                1 < v && (S[a++] = x[k++]))
                            } else {
                                k = a - E;
                                do {
                                    S[a++] = S[k++],
                                    S[a++] = S[k++],
                                    S[a++] = S[k++],
                                    v -= 3
                                } while (2 < v);
                                v && (S[a++] = S[k++],
                                1 < v && (S[a++] = S[k++]))
                            }
                            break
                        }
                    }
                    break
                }
            } while (n < i && a < s);
            return n -= v = p >> 3,
            d &= (1 << (p -= v << 3)) - 1,
            e.next_in = n,
            e.next_out = a,
            e.avail_in = n < i ? i - n + 5 : 5 - (n - i),
            e.avail_out = a < s ? s - a + 257 : 257 - (a - s),
            r.hold = d,
            void (r.bits = p)
        }
    }
    , function(e, t, r) {
        "use strict";
        var n = r(1)
          , i = 15
          , a = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0]
          , o = [16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 16, 72, 78]
          , s = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0]
          , l = [16, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 64, 64];
        e.exports = function(e, t, r, c, u, h, f, d) {
            var p, _, m, w, b, g, y, v, E, k = d.bits, x = 0, A = 0, S = 0, T = 0, I = 0, C = 0, B = 0, R = 0, D = 0, z = 0, N = null, P = 0, O = new n.Buf16(16), F = new n.Buf16(16), L = null, U = 0;
            for (x = 0; x <= i; x++)
                O[x] = 0;
            for (A = 0; A < c; A++)
                O[t[r + A]]++;
            for (I = k,
            T = i; 1 <= T && 0 === O[T]; T--)
                ;
            if (I > T && (I = T),
            0 == T)
                return u[h++] = 20971520,
                u[h++] = 20971520,
                d.bits = 1,
                0;
            for (S = 1; S < T && 0 === O[S]; S++)
                ;
            for (I < S && (I = S),
            R = 1,
            x = 1; x <= i; x++)
                if (R <<= 1,
                0 > (R -= O[x]))
                    return -1;
            if (0 < R && (0 === e || 1 != T))
                return -1;
            for (F[1] = 0,
            x = 1; x < i; x++)
                F[x + 1] = F[x] + O[x];
            for (A = 0; A < c; A++)
                0 !== t[r + A] && (f[F[t[r + A]]++] = A);
            if (0 === e ? (N = L = f,
            g = 19) : 1 === e ? (N = a,
            P -= 257,
            L = o,
            U -= 257,
            g = 256) : (N = s,
            L = l,
            g = -1),
            z = 0,
            A = 0,
            x = S,
            b = h,
            C = I,
            B = 0,
            m = -1,
            w = (D = 1 << I) - 1,
            1 === e && D > 852 || 2 === e && D > 592)
                return 1;
            for (; ; ) {
                y = x - B,
                f[A] < g ? (v = 0,
                E = f[A]) : f[A] > g ? (v = L[U + f[A]],
                E = N[P + f[A]]) : (v = 96,
                E = 0),
                p = 1 << x - B,
                S = _ = 1 << C;
                do {
                    u[b + (z >> B) + (_ -= p)] = y << 24 | v << 16 | E | 0
                } while (0 !== _);
                for (p = 1 << x - 1; z & p; )
                    p >>= 1;
                if (0 === p ? z = 0 : (z &= p - 1,
                z += p),
                A++,
                0 == --O[x]) {
                    if (x === T)
                        break;
                    x = t[r + f[A]]
                }
                if (x > I && (z & w) !== m) {
                    for (0 == B && (B = I),
                    b += S,
                    R = 1 << (C = x - B); C + B < T && !(0 >= (R -= O[C + B])); )
                        C++,
                        R <<= 1;
                    if (D += 1 << C,
                    1 === e && D > 852 || 2 === e && D > 592)
                        return 1;
                    u[m = z & w] = I << 24 | C << 16 | b - h | 0
                }
            }
            return 0 != z && (u[b + z] = 4194304 | x - B << 24),
            d.bits = I,
            0
        }
    }
    , function(e) {
        "use strict";
        e.exports = function() {
            this.text = 0,
            this.time = 0,
            this.xflags = 0,
            this.os = 0,
            this.extra = null,
            this.extra_len = 0,
            this.name = "",
            this.comment = "",
            this.hcrc = 0,
            this.done = !1
        }
    }
    , function(e, t, r) {
        "use strict";
        function n(e) {
            switch (e) {
            case "raw":
                return new P;
            case "eightbit":
                return new N;
            default:
                throw new Error("Unknown weight encoding")
            }
        }
        async function i(e, t, r) {
            let n;
            if (!(n = "string" == typeof (e = "string" == typeof e ? t(e) + (r && r.ignoreCache ? "?t=" + Date.now() : "") : Object.assign({}, e, {
                url: t(e.url) + (r && r.ignoreCache ? "?t=" + Date.now() : "")
            })) && function() {
                if (!window.hasOwnProperty("ProgressEvent") || !window.hasOwnProperty("FormData"))
                    return !1;
                let e = new XMLHttpRequest;
                if ("string" != typeof e.responseType)
                    return !1;
                try {
                    return e.responseType = "blob",
                    "blob" === e.responseType
                } catch (e) {
                    return !1
                }
            }() ? await function(e, t) {
                return new Promise(function(r, n) {
                    let i = new XMLHttpRequest;
                    i.open("GET", e, !0),
                    i.responseType = "blob";
                    let a = new F;
                    i.onload = function() {
                        a.forceDispatch();
                        let e = new Response(i.response);
                        r(e)
                    }
                    ,
                    i.onprogress = function(e) {
                        t && a.request(function() {
                            return t(e.loaded, e.total)
                        })
                    }
                    ,
                    i.onerror = function(e) {
                        n(e)
                    }
                    ,
                    i.send(null)
                }
                )
            }(e, r && r.progressCallback) : await fetch(e, r)).ok)
                throw new Error(`Fetch returns status code ${n.status}: ${n.statusText}`);
            return n
        }
        function a(e, t) {
            if (!t || !e.body)
                return e.arrayBuffer();
            let r = e.headers.get("Content-Length");
            if (!r)
                return e.arrayBuffer();
            const n = parseInt(r);
            let i = new Uint8Array(n)
              , a = 0
              , o = e.body.getReader()
              , s = new F;
            return o.read().then(function e(r) {
                return i.set(r.value, a),
                a += r.value.length,
                t && s.request(()=>t(a, n)),
                a == n ? (s.forceDispatch(),
                i.buffer) : o.read().then(e)
            })
        }
        function o(e=10) {
            return new Promise(t=>setTimeout(t, e))
        }
        function s(e) {
            return "WebGL2RenderingContext" === e.constructor.name
        }
        function l(e) {
            if (null === e)
                throw Error("Null is detected");
            return e
        }
        function c(e) {
            let t = e.getContext("2d");
            if (!t)
                throw Error("CanvasRenderingContext2D initialization failed");
            return t
        }
        function u(e, t={}) {
            let {srcX: r=0, srcY: n=0, srcW: i=e.width, srcH: a=e.height, dstX: o=0, dstY: s=0} = t
              , {dstW: l=i, dstH: u=a} = t
              , h = document.createElement("canvas");
            h.width = i,
            h.height = a,
            c(h).putImageData(e, -r, -n);
            let f = document.createElement("canvas");
            f.width = o + l,
            f.height = s + u;
            let d = c(f);
            return d.drawImage(h, 0, 0, i, a, o, s, l, u),
            d.getImageData(0, 0, o + l, s + u)
        }
        function h(e, t={}) {
            if (e instanceof HTMLCanvasElement)
                return function(e, t={}) {
                    let {srcX: r=0, srcY: n=0, srcW: i=e.width, srcH: a=e.height, dstX: o=0, dstY: s=0} = t
                      , {dstW: l=i, dstH: h=a} = t
                      , f = c(e).getImageData(r, n, i, a);
                    return (0 !== o || 0 !== s || i !== l || a !== h) && (f = u(f, {
                        dstX: o,
                        dstY: s,
                        dstW: l,
                        dstH: h
                    })),
                    f
                }(e, t);
            if (e instanceof HTMLVideoElement || e instanceof HTMLImageElement)
                return function(e, t={}) {
                    let r, n;
                    if (e instanceof HTMLVideoElement)
                        r = e.videoWidth,
                        n = e.videoHeight;
                    else {
                        if (!(e instanceof HTMLImageElement))
                            throw TypeError('Failed to execute "getImageDataFromDrawable(drawable, options)": "drawable" must be an instanceof HTMLVideoElement or HTMLImageElement');
                        r = e.naturalWidth,
                        n = e.naturalHeight
                    }
                    let {srcX: i=0, srcY: a=0, dstX: o=0, dstY: s=0, dstW: l=r, dstH: u=n} = t
                      , h = document.createElement("canvas");
                    h.width = o + l,
                    h.height = s + u;
                    let f = c(h);
                    return f.drawImage(e, i, a, r, n, o, s, l, u),
                    f.getImageData(0, 0, o + l, s + u)
                }(e, t);
            throw TypeError('Failed to execute "getImageData(image, options)": "image" must be an instance of HTMLCanvasElement, HTMLVideoElement, or HTMLImageElement')
        }
        function f(e, t, r={}) {
            let {srcX: n=0, srcY: i=0, srcW: a=e.width, srcH: o=e.height, dstX: s=0, dstY: l=0} = r
              , {dstW: h=a, dstH: f=o} = r;
            (0 !== n || 0 !== i || a !== h || o !== f) && (e = u(e, {
                srcX: n,
                srcY: i,
                srcW: a,
                srcH: o,
                dstW: h,
                dstH: f
            })),
            c(t).putImageData(e, s, l)
        }
        async function d(e) {
            let t = document.createElement("img");
            return new Promise((r,n)=>{
                t.onload = r,
                t.onerror = n,
                t.src = e
            }
            ).then(()=>t)
        }
        async function p(e) {
            let t = e.files;
            if (!t || 0 == t.length)
                throw new Error("No file is selected");
            return d(URL.createObjectURL(t[0]))
        }
        async function _() {
            let e = document.createElement("input");
            return e.type = "file",
            e.accept = "image/*",
            new Promise(t=>{
                e.onchange = (()=>t(p(e))),
                e.click()
            }
            )
        }
        function m(e) {
            if ("number" == typeof e)
                return [e, e, e, e];
            if (4 == e.length)
                return [e[0], e[1], e[2], e[3]];
            if (3 == e.length)
                return [e[0], e[1], e[2], e[0]];
            if (1 == e.length)
                return [e[0], e[0], e[0], e[0]];
            throw new Error("bias and scale must be scalar number or array of length 1 or 3 or 4.")
        }
        function w(e, t={}) {
            let {type: r=Float32Array, color: n=ne.RGB, order: i=re.HWC, bias: a=[0, 0, 0], scale: o=[1, 1, 1]} = t;
            const s = m(a)
              , l = m(o)
              , c = e.width
              , u = e.height;
            let h, f, d, p, _, w, b, g, y, v = e.data;
            switch (n) {
            case ne.RGB:
                switch (h = new r(c * u * 3),
                [w,b,g] = l,
                [f,d,p] = s,
                i) {
                case re.HWC:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[3 * (e * c + t) + 0] = (v[4 * (e * c + t) + 0] - f) / w,
                            h[3 * (e * c + t) + 1] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[3 * (e * c + t) + 2] = (v[4 * (e * c + t) + 2] - p) / g;
                    break;
                case re.CHW:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[(0 * u + e) * c + t] = (v[4 * (e * c + t) + 0] - f) / w,
                            h[(1 * u + e) * c + t] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[(2 * u + e) * c + t] = (v[4 * (e * c + t) + 2] - p) / g
                }
                break;
            case ne.BGR:
                switch (h = new r(c * u * 3),
                [f,d,p] = s,
                [w,b,g] = l,
                i) {
                case re.HWC:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[3 * (e * c + t) + 0] = (v[4 * (e * c + t) + 2] - p) / g,
                            h[3 * (e * c + t) + 1] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[3 * (e * c + t) + 2] = (v[4 * (e * c + t) + 0] - f) / w;
                    break;
                case re.CHW:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[(0 * u + e) * c + t] = (v[4 * (e * c + t) + 2] - p) / g,
                            h[(1 * u + e) * c + t] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[(2 * u + e) * c + t] = (v[4 * (e * c + t) + 0] - f) / w
                }
                break;
            case ne.RGBA:
                switch (h = new r(c * u * 4),
                [w,b,g,y] = l,
                [f,d,p,_] = s,
                i) {
                case re.HWC:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[4 * (e * c + t) + 0] = (v[4 * (e * c + t) + 0] - f) / w,
                            h[4 * (e * c + t) + 1] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[4 * (e * c + t) + 2] = (v[4 * (e * c + t) + 2] - p) / g,
                            h[4 * (e * c + t) + 3] = (v[4 * (e * c + t) + 3] - _) / y;
                    break;
                case re.CHW:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[(0 * u + e) * c + t] = (v[4 * (e * c + t) + 0] - f) / w,
                            h[(1 * u + e) * c + t] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[(2 * u + e) * c + t] = (v[4 * (e * c + t) + 2] - p) / g,
                            h[(3 * u + e) * c + t] = (v[4 * (e * c + t) + 3] - _) / y
                }
                break;
            case ne.BGRA:
                switch (h = new r(c * u * 4),
                [f,d,p,_] = s,
                [w,b,g,y] = l,
                i) {
                case re.HWC:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[4 * (e * c + t) + 0] = (v[4 * (e * c + t) + 2] - p) / g,
                            h[4 * (e * c + t) + 1] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[4 * (e * c + t) + 2] = (v[4 * (e * c + t) + 0] - f) / w,
                            h[4 * (e * c + t) + 3] = (v[4 * (e * c + t) + 3] - _) / y;
                    break;
                case re.CHW:
                    for (let e = 0; e < u; e++)
                        for (let t = 0; t < c; t++)
                            h[(0 * u + e) * c + t] = (v[4 * (e * c + t) + 2] - p) / g,
                            h[(1 * u + e) * c + t] = (v[4 * (e * c + t) + 1] - d) / b,
                            h[(2 * u + e) * c + t] = (v[4 * (e * c + t) + 0] - f) / w,
                            h[(3 * u + e) * c + t] = (v[4 * (e * c + t) + 3] - _) / y
                }
                break;
            case ne.GREY:
                h = new r(c * u),
                [w,b,g] = l,
                [f,d,p] = s;
                for (let e = 0; e < u; e++)
                    for (let t = 0; t < c; t++) {
                        let r = v[4 * (e * c + t) + 0]
                          , n = v[4 * (e * c + t) + 1]
                          , i = v[4 * (e * c + t) + 2];
                        h[e * c + t] = .2126 * (r - f) / w + .7162 * (n - d) / b + .0722 * (i - p) / g
                    }
                break;
            default:
                throw Error(`Unknown color format: ${n}`)
            }
            return h
        }
        function b(e, t={}) {
            let {type: r=Float32Array, color: n=ne.RGB, order: i=re.HWC, srcX: a=0, srcY: o=0, srcW: s=e.width, srcH: l=e.height, dstX: c=0, dstY: u=0, bias: f=[0, 0, 0], scale: d=[1, 1, 1]} = t
              , {dstW: p=s, dstH: _=l} = t;
            return w(h(e, {
                srcX: a,
                srcY: o,
                srcW: s,
                srcH: l,
                dstX: c,
                dstY: u,
                dstW: p,
                dstH: _
            }), {
                type: r,
                color: n,
                order: i,
                bias: f,
                scale: d
            })
        }
        function g(e, t={}) {
            let r, n;
            if (e instanceof HTMLVideoElement)
                r = e.videoWidth,
                n = e.videoHeight;
            else {
                if (!(e instanceof HTMLImageElement)) {
                    if (e instanceof HTMLCanvasElement)
                        return b(e, t);
                    if (e instanceof ImageData) {
                        let r = document.createElement("canvas");
                        return r.height = e.height,
                        r.width = e.width,
                        c(r).putImageData(e, 0, 0),
                        b(r, t)
                    }
                    throw TypeError('Failed to execute "getImageDataFromDrawable(drawable, options)": "drawable" must be an instanceof Drawable')
                }
                r = e.naturalWidth,
                n = e.naturalHeight
            }
            let {type: i=Float32Array, color: a=ne.RGB, order: o=re.HWC, srcX: s=0, srcY: l=0, dstX: u=0, dstY: h=0, dstW: f=r, dstH: d=n, bias: p=[0, 0, 0], scale: _=[1, 1, 1]} = t
              , m = document.createElement("canvas");
            return m.width = u + f,
            m.height = h + d,
            c(m).drawImage(e, s, l, r, n, u, h, f, d),
            b(m, {
                type: i,
                color: a,
                order: o,
                bias: p,
                scale: _
            })
        }
        async function y(e, t={}) {
            if ("string" == typeof e)
                return g(await d(e), t);
            if (e instanceof HTMLInputElement)
                return g(await p(e), t);
            if (e instanceof HTMLCanvasElement)
                return b(e, t);
            if (e instanceof HTMLImageElement || e instanceof HTMLVideoElement || e instanceof ImageData)
                return g(e, t);
            throw TypeError('Failed to execute "getImageData(image, options)": "image" must be an instance of string, HTMLInputElement, HTMLCanvasElement, HTMLImageElement, HTMLVideoElement, or ImageData object')
        }
        function v(e, t, r) {
            try {
                return new ImageData(e,t,r)
            } catch (n) {
                console.warn(`new ImageData failed: ${n}`);
                let i = c(document.createElement("canvas")).createImageData(t, r);
                return i.data.set(e),
                i
            }
        }
        function E(e, t, r, n, i={}) {
            let {color: a=ne.RGB, order: o=re.HWC, srcX: s=0, srcY: l=0, dstX: c=0, dstY: u=0, dstW: h=n.width, dstH: d=n.height, bias: p=[0, 0, 0], scale: _=[1, 1, 1]} = i;
            const w = m(p)
              , b = m(_);
            let g = t
              , y = r;
            e = function e(t) {
                return t instanceof Array ? Array.prototype.concat.apply([], t.map(t=>e(t))) : t
            }(e);
            let E, k, x, A, S, T, I, C, B = new Uint8ClampedArray(g * y * 4);
            switch (a) {
            case ne.RGB:
                switch ([E,k,x] = w,
                [S,T,I] = b,
                o) {
                case re.HWC:
                    for (let r = l; r < l + y; r++)
                        for (let n = s; n < s + g; n++)
                            B[4 * (r * t + n) + 0] = e[3 * (r * t + n) + 0] * S + E,
                            B[4 * (r * t + n) + 1] = e[3 * (r * t + n) + 1] * T + k,
                            B[4 * (r * t + n) + 2] = e[3 * (r * t + n) + 2] * I + x,
                            B[4 * (r * t + n) + 3] = 255;
                    break;
                case re.CHW:
                    for (let n = l; n < l + y; n++)
                        for (let i = s; i < s + g; i++)
                            B[4 * (n * t + i) + 0] = e[(0 * r + n) * t + i] * S + E,
                            B[4 * (n * t + i) + 1] = e[(1 * r + n) * t + i] * T + k,
                            B[4 * (n * t + i) + 2] = e[(2 * r + n) * t + i] * I + x,
                            B[4 * (n * t + i) + 3] = 255
                }
                break;
            case ne.BGR:
                switch ([E,k,x] = w,
                [S,T,I] = b,
                o) {
                case re.HWC:
                    for (let r = l; r < l + y; r++)
                        for (let n = s; n < s + g; n++)
                            B[4 * (r * t + n) + 0] = e[3 * (r * t + n) + 2] * S + E,
                            B[4 * (r * t + n) + 1] = e[3 * (r * t + n) + 1] * T + k,
                            B[4 * (r * t + n) + 2] = e[3 * (r * t + n) + 0] * I + x,
                            B[4 * (r * t + n) + 3] = 255;
                    break;
                case re.CHW:
                    for (let n = l; n < l + y; n++)
                        for (let i = s; i < s + g; i++)
                            B[4 * (n * t + i) + 0] = e[(2 * r + n) * t + i] * S + E,
                            B[4 * (n * t + i) + 1] = e[(1 * r + n) * t + i] * T + k,
                            B[4 * (n * t + i) + 2] = e[(0 * r + n) * t + i] * I + x,
                            B[4 * (n * t + i) + 3] = 255
                }
                break;
            case ne.RGBA:
                switch ([E,k,x,A] = w,
                [S,T,I,C] = b,
                o) {
                case re.HWC:
                    for (let r = l; r < l + y; r++)
                        for (let n = s; n < s + g; n++)
                            B[4 * (r * t + n) + 0] = e[3 * (r * t + n) + 0] * S + E,
                            B[4 * (r * t + n) + 1] = e[3 * (r * t + n) + 1] * T + k,
                            B[4 * (r * t + n) + 2] = e[3 * (r * t + n) + 2] * I + x,
                            B[4 * (r * t + n) + 3] = e[3 * (r * t + n) + 3] * C + A;
                    break;
                case re.CHW:
                    for (let n = l; n < l + y; n++)
                        for (let i = s; i < s + g; i++)
                            B[4 * (n * t + i) + 0] = e[(0 * r + n) * t + i] * S + E,
                            B[4 * (n * t + i) + 1] = e[(1 * r + n) * t + i] * T + k,
                            B[4 * (n * t + i) + 2] = e[(2 * r + n) * t + i] * I + x,
                            B[4 * (n * t + i) + 3] = e[(3 * r + n) * t + i] * C + A
                }
                break;
            case ne.BGRA:
                switch ([E,k,x,A] = w,
                [S,T,I,C] = b,
                o) {
                case re.HWC:
                    for (let r = l; r < l + y; r++)
                        for (let n = s; n < s + g; n++)
                            B[4 * (r * t + n) + 0] = e[4 * (r * t + n) + 2] * S + E,
                            B[4 * (r * t + n) + 1] = e[4 * (r * t + n) + 1] * T + k,
                            B[4 * (r * t + n) + 2] = e[4 * (r * t + n) + 0] * I + x,
                            B[4 * (r * t + n) + 3] = e[4 * (r * t + n) + 3] * C + A;
                    break;
                case re.CHW:
                    for (let n = l; n < l + y; n++)
                        for (let i = s; i < s + g; i++)
                            B[4 * (n * t + i) + 0] = e[(2 * r + n) * t + i] * S + E,
                            B[4 * (n * t + i) + 1] = e[(1 * r + n) * t + i] * T + k,
                            B[4 * (n * t + i) + 2] = e[(0 * r + n) * t + i] * I + x,
                            B[4 * (n * t + i) + 3] = e[(3 * r + n) * t + i] * C + A
                }
                break;
            case ne.GREY:
                for (let r = l; r < l + y; r++)
                    for (let n = s; n < s + g; n++)
                        B[4 * (r * t + n) + 0] = B[4 * (r * t + n) + 1] = B[4 * (r * t + n) + 2] = e[r * t + n] * _[0] + p[0],
                        B[4 * (r * t + n) + 3] = 255
            }
            f(v(B, g, y), n, {
                srcX: s,
                srcY: l,
                srcW: g,
                srcH: y,
                dstX: c,
                dstY: u,
                dstW: h,
                dstH: d
            })
        }
        function k(e, t=1) {
            let r = [[0, (e = e.slice()).length]]
              , n = [];
            for (let t = 0; t < e.length; t++)
                n[t] = t;
            for (; 0 < r.length; ) {
                let i, [a,o] = r.pop(), s = e[o - 1], l = a, c = o - 2;
                if (!(a >= o - 1)) {
                    for (; ; ) {
                        for (; e[l] > s && l <= c; )
                            l++;
                        for (; e[c] <= s && l <= c; )
                            c--;
                        if (l >= c)
                            break;
                        i = e[l],
                        e[l] = e[c],
                        e[c] = i,
                        i = n[l],
                        n[l] = n[c],
                        n[c] = i
                    }
                    e[o - 1] = e[l],
                    e[l] = s,
                    i = n[o - 1],
                    n[o - 1] = n[l],
                    n[l] = i,
                    r.push([a, l]),
                    l + 1 < t && r.push([l + 1, o])
                }
            }
            let i = [];
            for (let e = 0; e < t; e++)
                i.push(n[e]);
            return i
        }
        function x(e, t=1) {
            let r = [[0, (e = e.slice()).length]]
              , n = [];
            for (let t = 0; t < e.length; t++)
                n[t] = t;
            for (; 0 < r.length; ) {
                let i, [a,o] = r.pop(), s = e[o - 1], l = a, c = o - 2;
                if (!(a >= o - 1)) {
                    for (; ; ) {
                        for (; e[l] < s && l <= c; )
                            l++;
                        for (; e[c] >= s && l <= c; )
                            c--;
                        if (l >= c)
                            break;
                        i = e[l],
                        e[l] = e[c],
                        e[c] = i,
                        i = n[l],
                        n[l] = n[c],
                        n[c] = i
                    }
                    e[o - 1] = e[l],
                    e[l] = s,
                    i = n[o - 1],
                    n[o - 1] = n[l],
                    n[l] = i,
                    r.push([a, l]),
                    l + 1 < t && r.push([l + 1, o])
                }
            }
            let i = [];
            for (let e = 0; e < t; e++)
                i.push(n[e]);
            return i
        }
        function A(e, t) {
            return e in ie ? ie[e] : t
        }
        function S(e, t) {
            ie[e] = t
        }
        function T() {
            let e = {
                webgpu: ae.webgpu.checkAvailability(),
                webgl: ae.webgl.checkAvailability(),
                webassembly: ae.webassembly.checkAvailability(),
                fallback: ae.fallback.checkAvailability()
            }
              , t = ["webgpu", "webgl", "webassembly", "fallback"].filter(t=>e[t]);
            return {
                status: e,
                defaultOrder: t
            }
        }
        async function I(e, t) {
            if (!(e in ae))
                throw new Error(`Unknown backend: "${e}"`);
            let r;
            try {
                r = new ae[e](t),
                await r.init()
            } catch (t) {
                return console.warn(`Failed to initialize ${e} backend: ${t}`),
                null
            }
            return r
        }
        async function C(e, t={}) {
            let {backendOrder: r=null, backendOptions: n={}, cacheStrategy: i="latest", saveCache: a=!0, progressCallback: o, weightDirectory: s, transformUrlDelegate: l} = t;
            r || (r = T().defaultOrder),
            "string" == typeof r && (r = [r]),
            -1 === (r = r.slice()).indexOf("fallback") && r.concat(["fallback"]);
            for (let c = t=>(s && /\.bin/.test(t) && (t = t.replace(e, s)),
            l && (t = l(t)),
            t); 0 < r.length; ) {
                let s = r.shift()
                  , l = Object.assign({}, n[s]);
                l.transformUrlDelegate = c;
                let u = await I(s, l);
                if (u) {
                    try {
                        let r, n, s, l;
                        switch (i) {
                        case "latest":
                            if (s = await u.fetchDescriptor(e).catch(()=>null),
                            (l = await u.restoreCachedDescriptor(e)) && s && l.converted_at === s.converted_at && (r = l,
                            n = await u.restoreCachedParameters(e, o)))
                                break;
                            if (s && (r = s,
                            n = await u.fetchParameters(e, o)))
                                break;
                            if (l && (r = l,
                            n = await u.restoreCachedParameters(e, o)))
                                break;
                            throw Error("Network error is occurred and no cache is exist.");
                        case "networkOnly":
                        case "networkFirst":
                            if ((s = await u.fetchDescriptor(e).catch(()=>null)) && (r = s,
                            n = await u.fetchParameters(e, o)))
                                break;
                            if ("networkOnly" === i)
                                throw Error('Network error is occurred in "networkOnly" cache strategy');
                            if ((l = await u.restoreCachedDescriptor(e)) && (r = l,
                            n = await u.restoreCachedParameters(e, o)))
                                break;
                            throw Error("Network error is occurred and no cache is exist.");
                        case "cacheOnly":
                        case "cacheFirst":
                            if ((l = await u.restoreCachedDescriptor(e)) && (r = l,
                            n = await u.restoreCachedParameters(e, o)))
                                break;
                            if ("cacheOnly" === i)
                                throw Error('No cache is exist in "cacheOnly" cache strategy');
                            if ((s = await u.fetchDescriptor(e).catch(()=>null)) && (r = s,
                            n = await u.fetchParameters(e, o)))
                                break;
                            throw Error("Network error is occurred and no cache is exist.");
                        default:
                            throw Error(`"${i}" is not valid cache strategy name: "latest", "networkFirst", "networkOnly", "cacheFirst", "cacheOnly" is available.`)
                        }
                        if (a)
                            try {
                                await u.saveCache(e, r, n)
                            } catch (t) {}
                        await u.setDescriptorAndParameters(r, n)
                    } catch (e) {
                        console.warn(`Model loading failed for ${r} backend. Trying next backend: ${e.message}`);
                        continue
                    }
                    return u
                }
            }
            throw new Error("No backend is available")
        }
        r.r(t);
        var B = {};
        r.r(B),
        r.d(B, "Order", function() {
            return re
        }),
        r.d(B, "Color", function() {
            return ne
        }),
        r.d(B, "getImageArrayFromImageData", function() {
            return w
        }),
        r.d(B, "getImageArrayFromCanvas", function() {
            return b
        }),
        r.d(B, "getImageArrayFromDrawable", function() {
            return g
        }),
        r.d(B, "getImageArray", function() {
            return y
        }),
        r.d(B, "setImageArrayToCanvas", function() {
            return E
        }),
        r.d(B, "loadImageByUrl", function() {
            return d
        }),
        r.d(B, "loadImageFromFileInput", function() {
            return p
        }),
        r.d(B, "loadImageByDialog", function() {
            return _
        });
        var R = {};
        r.r(R),
        r.d(R, "argmax", function() {
            return k
        }),
        r.d(R, "argmin", function() {
            return x
        });
        var D = r(0)
          , z = r(9);
        class N {
            async decode(e) {
                let t = []
                  , r = 0
                  , n = new DataView(e.buffer,e.byteOffset)
                  , i = 0;
                for (; i < e.length; ) {
                    n.getInt32(i, !0),
                    i += 4;
                    let a = n.getInt32(i, !0);
                    i += 4;
                    let o = n.getFloat32(i, !0);
                    i += 8;
                    let s = new Float32Array(256);
                    for (let e = 0; 256 > e; e++)
                        s[e] = N.decode_table[127 & e] * o * (128 > e ? 1 : -1);
                    let l = new Uint8Array(e.buffer,e.byteOffset + i,a)
                      , c = z.inflate(l)
                      , u = c.length
                      , h = new Float32Array(u);
                    for (let e = 0; e < u; e++)
                        h[e] = s[c[e]];
                    t.push(h),
                    r += u,
                    i += a
                }
                let a = new Float32Array(r)
                  , o = 0;
                for (let e = 0; e < t.length; e++)
                    a.set(t[e], o),
                    o += t[e].length;
                return a
            }
        }
        N.decode_table = [0, 2750000021e-15, 7249999726e-15, 1875000089e-14, 3624999954e-14, 5874999624e-14, 8624999464e-14, .0001437500032, .0002312500001, .0003187500115, .0004062500084, .0005187499919, .0006562499912, .0007937499322, .0009312499315, .001218750025, .00165624998, .002093750052, .002531250007, .002968749963, .003406249918, .003843750106, .004281249829, .004843750037, .005531250034, .006218749564, .00690624956, .007593749557, .008281249553, .008968749084, .009656248614, .01109374966, .01328125037, .01546875015, .01765624993, .0198437497, .02203124948, .02421874925, .02640625089, .02859375067, .03078125045, .03296874836, .03515625, .03734375164, .03953124955, .04171875119, .04390624911, .04671875015, .0501562506, .05359374732, .05703124776, .06046874821, .06390624493, .06734374911, .07078124583, .07421874255, .07765624672, .08109374344, .08453124017, .08796874434, .09140624106, .09484373778, .09828124195, .10546875, .116406247, .127343744, .138281256, .149218753, .16015625, .171093747, .182031244, .192968756, .203906253, .21484375, .225781247, .236718744, .247656256, .2585937381, .26953125, .2804687619, .291406244, .302343756, .3132812381, .32421875, .3351562619, .346093744, .357031256, .3679687381, .37890625, .3898437619, .400781244, .411718756, .4226562381, .43359375, .4445312619, .458593756, .4757812321, .4929687381, .5101562142, .52734375, .5445312262, .5617187023, .5789062381, .5960937142, .61328125, .6304687262, .6476562023, .6648437381, .6820312142, .6992186904, .7164062262, .7335937023, .7507811785, .7679687142, .7851561904, .8023436666, .8195312023, .8367186785, .8539061546, .8710936904, .8882811666, .9054686427, .9226561785, .9398436546, .9570311308, .9742186666, .9914061427, 1];
        class P {
            async decode(e) {
                return new Float32Array(e.buffer,e.byteOffset,e.byteLength / 4)
            }
        }
        const O = -1;
        class F {
            constructor() {
                this.scheduledCallbackId = O
            }
            request(e) {
                this.fn = e,
                this.scheduledCallbackId == O && (this.scheduledCallbackId = requestAnimationFrame(()=>this.forceDispatch()))
            }
            forceDispatch() {
                this.scheduledCallbackId == O || (this.cancel(),
                this.fn())
            }
            cancel() {
                this.scheduledCallbackId == O || (cancelAnimationFrame(this.scheduledCallbackId),
                this.scheduledCallbackId = O)
            }
        }
        var L = r(2);
        class U {
            constructor(e=null, t=0, r, n=null) {
                if (this.placeholderContext = n,
                this._byteOffset = t,
                this._buffer = e,
                e)
                    this._length = void 0 === r ? e.byteLength / this.BYTES_PER_ELEMENT : r;
                else {
                    if (void 0 === r)
                        throw Error('"butter" or "length" must be specified.');
                    this._length = r
                }
                if (this.isDynamic && !n)
                    throw Error("PlaceholderContext must be required when SymbolicTypedArray is initialized as dynamic buffer view.")
            }
            get buffer() {
                return this._buffer || (this._buffer = new ArrayBuffer(this.byteOffset + this.byteLength)),
                this._buffer
            }
            set buffer(e) {
                this._buffer = e
            }
            get byteLength() {
                return this.length * this.BYTES_PER_ELEMENT
            }
            get offset() {
                return this.byteOffset / this.BYTES_PER_ELEMENT
            }
            get isDynamic() {
                return "number" != typeof this._byteOffset || "number" != typeof this._length
            }
            get length() {
                return this.isDynamic ? this.placeholderContext.resolve(this._length) : this._length
            }
            get byteOffset() {
                return this.isDynamic ? this.placeholderContext.resolve(this._byteOffset) : this._byteOffset
            }
            copyWithin(e, t, r) {
                return this.toActual().copyWithin(e, t, r),
                this
            }
            fill(e, t, r) {
                return this.toActual().fill(e, t, r),
                this
            }
            indexOf(e, t) {
                return this.toActual().indexOf(e, t)
            }
            join(e) {
                return this.toActual().join(e)
            }
            lastIndexOf(e, t) {
                return this.toActual().lastIndexOf(e, t)
            }
            sort(e) {
                return this.toActual().sort(e),
                this
            }
            includes(e, t) {
                return this.toActual().includes(e, t)
            }
            set(e, t) {
                return this.toActual().set(function e(t) {
                    let r = [];
                    for (let n, i = 0; i < t.length; i++)
                        (n = t[i])instanceof Array ? r.splice(r.length, 0, e(n)) : r[r.length] = n;
                    return r
                }(e), t)
            }
            toLocaleString() {
                return this.toActual().toLocaleString()
            }
            toString() {
                return this.toActual().toString()
            }
            [Symbol.iterator]() {
                return this.toActual()[Symbol.iterator]()
            }
            entries() {
                return this.toActual().entries()
            }
            keys() {
                return this.toActual().keys()
            }
            values() {
                return this.toActual().values()
            }
        }
        class M extends U {
            constructor() {
                super(...arguments),
                this.BYTES_PER_ELEMENT = 4
            }
            toActual() {
                if (!this.buffer)
                    throw new Error("Internal buffer for this variable is not set. DescriptorRunner.setPlaceholderValue() have to be called before calling this function.");
                return new Float32Array(this.buffer,this.byteOffset,this.length)
            }
            every(e, t) {
                return this.toActual().every(e, t)
            }
            filter(e, t) {
                return this.toActual().filter(e, t)
            }
            find(e, t) {
                return this.toActual().find(e, t)
            }
            findIndex(e, t) {
                return this.toActual().findIndex(e, t)
            }
            forEach(e, t) {
                return this.toActual().forEach(e, t)
            }
            map(e, t) {
                return this.toActual().map(e, t)
            }
            reduce(e, t) {
                return this.toActual().reduce(e, t)
            }
            reduceRight(e, t) {
                return this.toActual().reduceRight(e, t)
            }
            reverse() {
                return this.toActual().reverse()
            }
            slice(e, t) {
                return this.toActual().slice(e, t)
            }
            some(e, t) {
                return this.toActual().some(e, t)
            }
            subarray(e, t) {
                return this.toActual().subarray(e, t)
            }
            includes(e, t) {
                return this.toActual().includes(e, t)
            }
        }
        M.BYTES_PER_ELEMENT = 4;
        class j {
            constructor(e={}) {
                this.descriptor = null;
                let {transformUrlDelegate: t=function(e) {
                    return e
                }
                } = e;
                this.transformUrlDelegate = t
            }
            static checkAvailability() {
                return !1
            }
        }
        class W extends j {
            constructor(e={}) {
                if (super(e),
                this.backendName = "webassembly",
                this.worker_promise_reject_func = null,
                this.worker_initial_error = null,
                "undefined" == typeof Worker)
                    throw new Error("WebWorker is needed for WebAssembly backend");
                "object" != typeof WebAssembly && console.warn("WebAssembly is not supported on this browser, trying to use asm.js code")
            }
            static checkAvailability() {
                return "Worker"in window
            }
            init() {
                if (!W.checkAvailability())
                    throw Error("WebAssembly backend is not supported in this browser.");
                return Promise.resolve()
            }
            absolutePath(e) {
                var t = document.createElement("span");
                return t.insertAdjacentHTML("beforeend", '<a href="' + e + '" />'),
                t.firstChild.href
            }
            async setDescriptorAndParameters(e, t) {
                this.descriptor = e,
                this.placeholderContext = new L.a(this.descriptor.placeholders);
                let r = "object" == typeof WebAssembly ? "webassembly" : "asmjs";
                0 <= window.navigator.userAgent.indexOf("iPhone OS 11_2") && (r = "asmjs");
                let n = `${this.directory}/kernels_${r}.js`;
                n = this.transformUrlDelegate(n),
                this.worker_entry_js_path = n;
                let i = await fetch(this.worker_entry_js_path)
                  , a = await i.text()
                  , o = (e,t)=>{
                    let r = this.absolutePath(`${this.directory}/${e}`)
                      , n = this.transformUrlDelegate(r);
                    a = a.replace(t, n)
                }
                ;
                "webassembly" == r ? o("kernels_webassembly.wasm", "WEBDNN_URL_KERNELS_WASM") : o("kernels_asmjs.js.mem", "WEBDNN_URL_KERNELS_ASMJS_MEM"),
                await this.compile(a),
                await this.loadWeights(new Uint8Array(t)),
                (await this.getInputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                    e.buffer = new Float32Array(e.length).buffer
                }
                ),
                (await this.getOutputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                    e.buffer = new Float32Array(e.length).buffer
                }
                )
            }
            async fetchDescriptor(e) {
                return this.directory = e,
                (await i(`${e}/graph_${this.backendName}.json`, this.transformUrlDelegate)).json()
            }
            async fetchParameters(e, t) {
                let r = `${e}/weight_${this.backendName}.bin`;
                return a(await i(r, this.transformUrlDelegate), t)
            }
            async restoreCachedDescriptor(e) {
                return this.directory = e,
                D.getItem(`${e}_${this.backendName}_descriptor`).catch(()=>null)
            }
            async restoreCachedParameters(e, t) {
                let r = await D.getItem(`${e}_${this.backendName}_parameters`).catch(()=>null);
                return r && t && t(r.byteLength, r.byteLength),
                r
            }
            async saveCache(e, t, r) {
                await Promise.all([D.setItem(`${e}_${this.backendName}_descriptor`, t), D.setItem(`${e}_${this.backendName}_parameters`, r)])
            }
            async setPlaceholderValue(e) {
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized.");
                let t = this.placeholderContext;
                if (t.update(e),
                !t.isResolved)
                    return;
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                let r = this.descriptor.unresolved_value_lists
                  , n = [];
                for (let e, i = 0; i < r.length; i++)
                    (e = r[i]).forEach(e=>{
                        let r = t.resolve(e.placeholder);
                        n.push(i, e.offset, r)
                    }
                    );
                (await this.getInputViews()).filter(e=>e.isDynamic).forEach(e=>{
                    e.buffer = new Float32Array(e.length).buffer
                }
                ),
                (await this.getOutputViews()).filter(e=>e.isDynamic).forEach(e=>{
                    e.buffer = new Float32Array(e.length).buffer
                }
                );
                let i = this.placeholderContext.resolve(this.descriptor.memory_layout.dynamic.size);
                await this.setPlaceholderValueWorker(i, new Int32Array(n))
            }
            setPlaceholderValueWorker(e, t) {
                if (!this.worker)
                    throw Error("Worker is not initialized");
                let r = this.worker;
                return new Promise((n,i)=>{
                    r.onmessage = (e=>{
                        0 === e.data ? n() : (console.log(e.data),
                        r.terminate(),
                        i(new Error(e.data)))
                    }
                    ),
                    r.postMessage({
                        type: "set_dynamic_buffer",
                        size: e,
                        data: t
                    })
                }
                )
            }
            compile(e) {
                let t = new Blob([e],{
                    type: "text/javascript"
                })
                  , r = URL.createObjectURL(t)
                  , n = new Worker(r);
                n.onerror = (e=>{
                    console.error(e),
                    this.worker_promise_reject_func ? this.worker_promise_reject_func(e) : this.worker_initial_error = e
                }
                );
                let i = new Promise((e,t)=>this.worker_initial_error ? t(this.worker_initial_error) : (this.worker_promise_reject_func = t,
                void (n.onmessage = (r=>{
                    0 === r.data ? e() : (console.error(r.data),
                    n.terminate(),
                    t(new Error(r.data)))
                }
                ))));
                return this.worker = n,
                i
            }
            async loadWeights(e) {
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.worker)
                    throw new Error("Worker is not initialized");
                let t = n(this.descriptor.weight_encoding)
                  , r = await t.decode(e)
                  , i = this.worker;
                return new Promise((e,t)=>{
                    this.worker_promise_reject_func = t,
                    i.onmessage = (r=>{
                        0 === r.data ? e() : (console.log(r.data),
                        i.terminate(),
                        t(new Error(r.data)))
                    }
                    ),
                    i.postMessage({
                        type: "weight",
                        data: r
                    }, [r.buffer])
                }
                )
            }
            getInputViews() {
                if (this.inputs)
                    return this.inputs;
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized");
                let e = this.descriptor
                  , t = this.placeholderContext;
                return this.inputs = e.inputs.map(r=>{
                    let n = e.memory_layout.static.allocations[r] || e.memory_layout.dynamic.allocations[r];
                    return new M(null,0,n.size,t)
                }
                ),
                this.inputs
            }
            getOutputViews() {
                if (this.outputs)
                    return this.outputs;
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized");
                let e = this.descriptor
                  , t = this.placeholderContext;
                return this.outputs = e.outputs.map(r=>{
                    let n = e.memory_layout.static.allocations[r] || e.memory_layout.dynamic.allocations[r];
                    return new M(null,0,n.size,t)
                }
                ),
                this.outputs
            }
            async run() {
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.worker)
                    throw new Error("Worker is not initialized");
                if (!this.placeholderContext.isResolved)
                    throw new Error("Not all placeholder is resolved");
                let e = this.placeholderContext
                  , t = this.descriptor
                  , r = this.worker;
                return new Promise((n,i)=>{
                    this.worker_promise_reject_func = i,
                    r.onmessage = (e=>{
                        if (Array.isArray(e.data)) {
                            for (let t = 0; t < e.data.length; t++)
                                this.outputs[t].set(e.data[t]);
                            n()
                        } else
                            console.log(e.data),
                            r.terminate(),
                            i(new Error(e.data))
                    }
                    );
                    let a = [t.memory_layout.static.allocations, t.memory_layout.dynamic.allocations]
                      , o = [];
                    for (let r = 0; r < t.inputs.length; r++)
                        for (let n, i = 0; 2 > i; i++)
                            if (n = a[i][t.inputs[r]]) {
                                let t = this.inputs[r];
                                o.push({
                                    space: i,
                                    offset: e.resolve(n.offset),
                                    size: t.length,
                                    data: t.toActual()
                                });
                                break
                            }
                    let s = [];
                    for (let r = 0; r < t.outputs.length; r++)
                        for (let n, i = 0; 2 > i; i++)
                            if (n = a[i][t.outputs[r]]) {
                                let t = this.outputs[r];
                                s.push({
                                    space: i,
                                    offset: e.resolve(n.offset),
                                    size: t.length
                                });
                                break
                            }
                    r.postMessage({
                        type: "run",
                        inputs: o,
                        outputs: s
                    })
                }
                )
            }
        }
        let H;
        class V {
            constructor() {
                this.gl = l(V.initializeContext())
            }
            static getInstance() {
                return H || (H = new V),
                H
            }
            createTexture(e, t, r, n) {
                let i = this.gl
                  , a = l(i.createTexture());
                return i.activeTexture(i.TEXTURE0 + 9),
                i.bindTexture(i.TEXTURE_2D, a),
                i.texImage2D(i.TEXTURE_2D, 0, r, e, t, 0, n, i.FLOAT, null),
                i.texParameteri(i.TEXTURE_2D, i.TEXTURE_WRAP_S, i.CLAMP_TO_EDGE),
                i.texParameteri(i.TEXTURE_2D, i.TEXTURE_WRAP_T, i.CLAMP_TO_EDGE),
                i.texParameteri(i.TEXTURE_2D, i.TEXTURE_MIN_FILTER, i.NEAREST),
                i.texParameteri(i.TEXTURE_2D, i.TEXTURE_MAG_FILTER, i.NEAREST),
                i.bindTexture(i.TEXTURE_2D, null),
                a
            }
            createVertexShader(e) {
                return this.createShader(this.gl.VERTEX_SHADER, e)
            }
            createFragmentShader(e) {
                return this.createShader(this.gl.FRAGMENT_SHADER, e)
            }
            createShader(e, t) {
                let r = l(this.gl.createShader(e));
                if (this.gl.shaderSource(r, t),
                this.gl.compileShader(r),
                !this.gl.getShaderParameter(r, this.gl.COMPILE_STATUS))
                    throw console.error(this.gl.getShaderInfoLog(r)),
                    Error("Shader Compile failed: " + this.gl.getShaderInfoLog(r));
                return r
            }
            createProgram(e, t) {
                let r = l(this.gl.createProgram());
                if (this.gl.attachShader(r, t),
                this.gl.attachShader(r, e),
                this.gl.linkProgram(r),
                !this.gl.getProgramParameter(r, this.gl.LINK_STATUS))
                    throw console.error(this.gl.getProgramInfoLog(r)),
                    Error("ShaderProgram Initialization failed.");
                return r
            }
            createArrayBuffer(e) {
                let t = l(this.gl.createBuffer());
                return this.gl.bindBuffer(this.gl.ARRAY_BUFFER, t),
                this.gl.bufferData(this.gl.ARRAY_BUFFER, e, this.gl.STATIC_DRAW),
                t
            }
            createFrameBuffer() {
                return l(this.gl.createFramebuffer())
            }
            bindArrayBuffer(e) {
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, e)
            }
            bindFrameBuffer(e, t, r) {
                this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, e),
                this.gl.viewport(0, 0, t, r),
                this.gl.scissor(0, 0, t, r)
            }
            useProgram(e) {
                this.gl.useProgram(e)
            }
            deleteTexture(e) {
                this.gl.deleteTexture(e)
            }
            static initializeWebGL2Context(e=document.createElement("canvas")) {
                let t;
                return (t = e.getContext("webgl2")) && t.getExtension("EXT_color_buffer_float") ? A("DEBUG", !1) && !t.getExtension("WEBGL_debug_renderer_info") ? null : t : null
            }
            static initializeWebGL1Context(e=document.createElement("canvas")) {
                let t = e.getContext("webgl") || e.getContext("experimental-webgl");
                return t && t.getExtension("OES_texture_float") ? V.IS_SAFARI ? null : A("DEBUG", !1) && !t.getExtension("WEBGL_debug_renderer_info") ? null : t : null
            }
            static initializeContext() {
                let e, t = document.createElement("canvas");
                if (e = V.initializeWebGL2Context(t))
                    A("DEBUG", !1) && console.info("WebGL2 is enabled");
                else {
                    if (!(e = V.initializeWebGL1Context(t)))
                        return null;
                    A("DEBUG", !1) && console.info("WebGL2 is disabled")
                }
                return e.disable(e.DEPTH_TEST),
                e.disable(e.STENCIL_TEST),
                e.disable(e.BLEND),
                e.disable(e.DITHER),
                e.disable(e.POLYGON_OFFSET_FILL),
                e.disable(e.SAMPLE_COVERAGE),
                e.enable(e.SCISSOR_TEST),
                e.enable(e.CULL_FACE),
                e.cullFace(e.BACK),
                e
            }
            static checkAvailability() {
                if (null === $) {
                    let e = V.initializeContext();
                    $ = !(!e || 4096 > A("MAX_TEXTURE_SIZE", e.getParameter(e.MAX_TEXTURE_SIZE)))
                }
                return $
            }
            async waitForComplete() {
                let e = this.gl;
                if (s(e)) {
                    let t = e.fenceSync(e.SYNC_GPU_COMMANDS_COMPLETE, 0)
                      , r = e.clientWaitSync(t, 0, 0);
                    for (; r !== e.CONDITION_SATISFIED && r !== e.ALREADY_SIGNALED; )
                        await new Promise(e=>setTimeout(e, 1)),
                        r = e.clientWaitSync(t, 0, 0);
                    e.deleteSync(t)
                } else
                    e.finish()
            }
            get MAX_TEXTURE_SIZE() {
                let e = A("MAX_TEXTURE_SIZE", this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE));
                if (16384 <= e)
                    return 4096;
                if (8192 <= e)
                    return 4096;
                if (4096 <= e)
                    return 4096;
                throw new Error(`MAX_TEXTURE_SIZE is too small: ${e}`)
            }
        }
        V.IS_SAFARI = -1 !== navigator.userAgent.toLowerCase().indexOf("safari") && -1 === navigator.userAgent.toLowerCase().indexOf("chrome");
        let$ = null;
        class G {
            constructor(e, t) {
                this.byteLength = e,
                this.backend = t
            }
        }
        class X extends G {
            constructor(e, t, r, n, i, a) {
                switch (super(e, "webgl"),
                this._texture = null,
                this.readTextureUnitIndices = [],
                this.isBoundToDrawFrameBuffer = !1,
                this.handler = V.getInstance(),
                this.name = n,
                this.channelMode = a,
                a) {
                case "RGBA":
                    this.elementsPerPixel = 4;
                    break;
                case "R":
                    this.elementsPerPixel = 1;
                    break;
                default:
                    throw Error("Unknown channel mode")
                }
                if (s(this.handler.gl))
                    switch (a) {
                    case "RGBA":
                        this.textureFormat = this.handler.gl.RGBA,
                        this.textureInternalFormat = this.handler.gl.RGBA32F,
                        this.pixelStride = 4;
                        break;
                    case "R":
                        this.textureFormat = this.handler.gl.RED,
                        this.textureInternalFormat = this.handler.gl.R32F,
                        this.pixelStride = 1;
                        break;
                    default:
                        throw Error("Unknown channel mode")
                    }
                else
                    this.textureFormat = this.handler.gl.RGBA,
                    this.textureInternalFormat = this.handler.gl.RGBA,
                    this.pixelStride = 4;
                if (this.pixelStride < this.elementsPerPixel)
                    throw Error("elementsPerPixel must be smaller than pixelStride");
                this.array = i || new Float32Array(this.length),
                this.textureWidth = t,
                this.textureHeight = r
            }
            get texture() {
                return this._texture
            }
            get length() {
                return this.byteLength / Float32Array.BYTES_PER_ELEMENT
            }
            async write(e, t) {
                this.array.set(e, t),
                await this.syncWriteViews()
            }
            async read(e, t=0, r) {
                if (e !== Float32Array)
                    throw new Error("Currently, only Float32Array is supported for parameter 'dst'.");
                await this.syncReadViews(),
                new Float32Array(this.array.buffer,t * Float32Array.BYTES_PER_ELEMENT,r)
            }
            getWriteView(e, t, r) {
                return new r(this.array.buffer,e * r.BYTES_PER_ELEMENT,t)
            }
            getReadView(e, t, r) {
                return new r(this.array.buffer,e * r.BYTES_PER_ELEMENT,t)
            }
            async syncWriteViews() {
                let e = this.handler.gl;
                this.texture || this.allocateTexture();
                let t = this.pack(this.array);
                if (t.length != this.textureWidth * this.textureHeight * this.pixelStride) {
                    let e = new Float32Array(this.textureWidth * this.textureHeight * this.pixelStride);
                    e.set(t, 0),
                    t = e
                }
                await this.bindToReadTexture(9),
                e.texSubImage2D(e.TEXTURE_2D, 0, 0, 0, this.textureWidth, this.textureHeight, this.textureFormat, e.FLOAT, t),
                this.unbindFromReadTexture()
            }
            async syncReadViews() {
                let e = this.handler.gl;
                const t = e.RGBA;
                let r = new Float32Array(this.textureWidth * this.textureHeight * 4);
                this.bindToDrawTexture(),
                e.readPixels(0, 0, this.textureWidth, this.textureHeight, t, e.FLOAT, r),
                this.unbindFromDrawTexture(),
                r = this.unpack(r),
                this.array.set(r.slice(0, this.length), 0)
            }
            async bindToReadTexture(e) {
                if (this.isBoundToDrawFrameBuffer)
                    throw Error("This buffer is already registered as draw buffer. You may forgot to unbind the binding while previous operations.");
                let t = this.handler.gl;
                this.texture || (this.allocateTexture(),
                await this.syncWriteViews()),
                t.activeTexture(t.TEXTURE0 + e),
                t.bindTexture(t.TEXTURE_2D, this.texture),
                this.readTextureUnitIndices.push(e)
            }
            unbindFromReadTexture() {
                let e = this.handler.gl;
                for (let t of this.readTextureUnitIndices)
                    e.activeTexture(e.TEXTURE0 + t),
                    e.bindTexture(e.TEXTURE_2D, null);
                this.readTextureUnitIndices = []
            }
            bindToDrawTexture() {
                if (0 < this.readTextureUnitIndices.length)
                    throw Error("This buffer is already registered as read buffer. You cannot bind a texture as both read and draw texture buffer at same time.");
                if (this.isBoundToDrawFrameBuffer)
                    throw Error("This buffer is already registered as draw buffer. You may forgot to unbind the binding while previous operations.");
                let e = this.handler.gl;
                this.texture || this.allocateTexture(),
                e.framebufferTexture2D(e.FRAMEBUFFER, e.COLOR_ATTACHMENT0, e.TEXTURE_2D, this.texture, 0),
                this.isBoundToDrawFrameBuffer = !0
            }
            unbindFromDrawTexture() {
                if (this.isBoundToDrawFrameBuffer) {
                    let e = this.handler.gl;
                    e.framebufferTexture2D(e.FRAMEBUFFER, e.COLOR_ATTACHMENT0, e.TEXTURE_2D, null, 0),
                    this.isBoundToDrawFrameBuffer = !1
                }
            }
            pack(e) {
                let t = this.pixelStride / this.elementsPerPixel;
                if (1 == t)
                    return new Float32Array(e);
                {
                    let r = new Float32Array(e.length * t);
                    for (let n = 0; n < e.length; n++)
                        r[n * t] = e[n];
                    return r
                }
            }
            unpack(e) {
                let t = 4 / this.elementsPerPixel;
                if (1 == t)
                    return new Float32Array(e);
                {
                    let r = new Float32Array(e.length / t);
                    for (let n = 0; n < e.length / t; n++)
                        r[n] = e[n * t];
                    return r
                }
            }
            allocateTexture() {
                if (this.texture)
                    throw Error("Texture is already allocated.");
                this._texture = this.handler.createTexture(this.textureWidth, this.textureHeight, this.textureInternalFormat, this.textureFormat)
            }
        }
        const Y = new Float32Array([-1, 1, -1, -1, 1, 1, 1, -1]);
        class Z extends j {
            constructor(e={}) {
                super(e),
                this.backendName = "webgl"
            }
            static checkAvailability() {
                return V.checkAvailability()
            }
            async init() {
                if (!Z.checkAvailability())
                    throw Error("WebGL backend is not supported in this browser.");
                this.handler = V.getInstance();
                let e = this.handler.createArrayBuffer(Y);
                this.handler.bindArrayBuffer(e),
                this.buffers = new Map
            }
            async fetchDescriptor(e) {
                return (await i(`${e}/graph_${this.backendName}_${this.handler.MAX_TEXTURE_SIZE}.json`, this.transformUrlDelegate)).json()
            }
            async fetchParameters(e, t) {
                return a(await i(`${e}/weight_${this.backendName}_${this.handler.MAX_TEXTURE_SIZE}.bin`, this.transformUrlDelegate), t)
            }
            async restoreCachedDescriptor(e) {
                return D.getItem(`${e}_${this.backendName}_${this.handler.MAX_TEXTURE_SIZE}_descriptor`).catch(()=>null)
            }
            async restoreCachedParameters(e, t) {
                let r = await D.getItem(`${e}_${this.backendName}_${this.handler.MAX_TEXTURE_SIZE}_parameters`).catch(()=>null);
                return r && t && t(r.byteLength, r.byteLength),
                r
            }
            async saveCache(e, t, r) {
                await Promise.all([D.setItem(`${e}_${this.backendName}_${this.handler.MAX_TEXTURE_SIZE}_descriptor`, t), D.setItem(`${e}_${this.backendName}_${this.handler.MAX_TEXTURE_SIZE}_parameters`, r)])
            }
            async setDescriptorAndParameters(e, t) {
                await this.setDescriptor(e),
                await this.compile(),
                await this.initializeStaticBuffer(t),
                this.placeholderContext && this.placeholderContext.isResolved && await this.initializeDynamicBuffer()
            }
            async initializeStaticBuffer(e) {
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                let t = this.descriptor
                  , r = n(this.descriptor.weight_encoding)
                  , i = await r.decode(new Uint8Array(e))
                  , a = this.buffers
                  , o = t.memory_layout.mapping;
                Object.entries(t.memory_layout.static.allocations).forEach(([e,{width: t, height: r, size: n, channel_mode: i}])=>{
                    a.set(e, new X(n * Float32Array.BYTES_PER_ELEMENT,t,r,e,null,i))
                }
                ),
                Object.entries(t.constants_map).forEach(([e,{size: t, byte_offset: r}])=>{
                    a.get(e).array.set(new Float32Array(i.buffer,r,t))
                }
                ),
                (await this.getInputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                    e.buffer = a.get(o[e.name]).getWriteView(0, e.length, Float32Array).buffer
                }
                ),
                (await this.getOutputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                    e.buffer = a.get(o[e.name]).getReadView(0, e.length, Float32Array).buffer
                }
                )
            }
            async initializeDynamicBuffer() {
                if (!this.descriptor)
                    throw Error("GraphDescriptor is not loaded.");
                if (!this.placeholderContext)
                    throw Error("PlaceholderContext is not initialized.");
                let e = this.descriptor
                  , t = this.placeholderContext
                  , r = this.buffers
                  , n = e.memory_layout.mapping;
                Object.entries(e.memory_layout.dynamic.allocations).forEach(([e,{width: n, height: i, size: a, channel_mode: o}])=>{
                    r.set(e, new X(t.resolve(a) * Float32Array.BYTES_PER_ELEMENT,t.resolve(n),t.resolve(i),e,null,o))
                }
                ),
                (await this.getInputViews()).filter(e=>e.isDynamic).forEach(e=>{
                    e.buffer = r.get(n[e.name]).getWriteView(0, t.resolve(e.length), Float32Array).buffer
                }
                ),
                (await this.getOutputViews()).filter(e=>e.isDynamic).forEach(e=>{
                    e.buffer = r.get(n[e.name]).getReadView(0, t.resolve(e.length), Float32Array).buffer
                }
                ),
                this.buildPipeline()
            }
            async setDescriptor(e) {
                this.descriptor = e,
                this.placeholderContext = new L.a(e.placeholders)
            }
            async compile() {
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                let e = this.descriptor;
                this.programs = new Map,
                this.vertexShader = this.handler.createVertexShader("\n            precision highp float;\n            attribute vec2 _xy;\n            void main() { \n              gl_Position = vec4(_xy, 0, 1); \n            }\n        "),
                Object.keys(e.shader_sources).forEach(t=>{
                    let r = this.handler.createFragmentShader(e.shader_sources[t])
                      , n = this.handler.createProgram(this.vertexShader, r);
                    this.programs.set(t, n)
                }
                )
            }
            async setPlaceholderValue(e) {
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized.");
                let t = this.placeholderContext;
                if (t.update(e),
                t.isResolved) {
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (await this.initializeDynamicBuffer(),
                    0 < Object.keys(this.descriptor.placeholders).length)
                        throw Error("Currently, WebGL backend doesn't support Placeholder feature.")
                }
            }
            getInputViews() {
                if (this.inputs)
                    return this.inputs;
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized");
                let e = this.descriptor
                  , t = this.placeholderContext
                  , r = this.descriptor.memory_layout.mapping;
                return this.inputs = e.inputs.map(e=>{
                    let n = new M(null,0,this.buffers.get(r[e]).length,t);
                    return n.name = e,
                    n
                }
                ),
                this.inputs
            }
            getOutputViews() {
                if (this.outputs)
                    return this.outputs;
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized");
                let e = this.descriptor
                  , t = this.placeholderContext
                  , r = this.descriptor.memory_layout.mapping;
                return this.outputs = e.outputs.map(e=>{
                    let n = new M(null,0,this.buffers.get(r[e]).length,t);
                    return n.name = e,
                    n
                }
                ),
                this.outputs
            }
            buildPipeline() {
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized");
                if (!this.placeholderContext.isResolved)
                    throw new Error(`Not all placeholders are resolved: ${this.placeholderContext}`);
                let e = this.handler.gl
                  , t = this.buffers
                  , r = this.descriptor.memory_layout.mapping
                  , n = new Map;
                this.runtimeInfo = {
                    inputs: this.getInputViews().map(e=>t.get(r[e.name])),
                    outputs: this.getOutputViews().map(e=>t.get(r[e.name])),
                    programs: this.descriptor.exec_infos.map(i=>{
                        let a = i.inputs.map(e=>{
                            let i = t.get(r[e.variable_name]);
                            return n.has(i) || n.set(i, 0),
                            n.set(i, n.get(i) + 1),
                            {
                                buffer: i,
                                uniformIndex: e.value
                            }
                        }
                        )
                          , o = t.get(r[i.output])
                          , s = this.programs.get(i.shader_name);
                        this.handler.useProgram(s);
                        let l = Object.keys(i.uniforms).map(t=>{
                            let {type: r, value: n} = i.uniforms[t];
                            switch (r) {
                            case "int":
                                return {
                                    func: e.uniform1i,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "float":
                                return {
                                    func: e.uniform1f,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "vec2":
                                return {
                                    func: e.uniform2fv,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "vec3":
                                return {
                                    func: e.uniform3fv,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "vec4":
                                return {
                                    func: e.uniform4fv,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "ivec2":
                                return {
                                    func: e.uniform2iv,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "ivec3":
                                return {
                                    func: e.uniform3iv,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "ivec4":
                                return {
                                    func: e.uniform4iv,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            case "sampler2D":
                                return {
                                    func: e.uniform1i,
                                    args: [e.getUniformLocation(s, t), n]
                                };
                            default:
                                throw TypeError(`Incompatible type for uniform parameter: ${r}`)
                            }
                        }
                        )
                          , c = e.getAttribLocation(s, "_xy");
                        return {
                            program: s,
                            frameBuffer: this.handler.createFrameBuffer(),
                            name: i.shader_name,
                            width: o.textureWidth,
                            height: o.textureHeight,
                            inputs: a,
                            output: o,
                            xyAttribLoc: c,
                            uniforms: l,
                            disposable: []
                        }
                    }
                    )
                };
                for (let e of this.runtimeInfo.programs)
                    e.inputs.forEach(({buffer: t})=>{
                        let r = n.get(t) - 1;
                        0 == r && e.disposable.push(t),
                        n.set(t, r)
                    }
                    )
            }
            async run() {
                if (!this.descriptor)
                    throw new Error("Descriptor is not loaded");
                if (!this.placeholderContext)
                    throw new Error("PlaceholderContext is not initialized");
                if (!this.placeholderContext.isResolved)
                    throw new Error(`Not all placeholders are resolved: ${this.placeholderContext}`);
                let e = this.handler.gl
                  , t = this.runtimeInfo;
                if (0 < this.runtimeInfo.programs.length) {
                    for (let e of t.inputs)
                        await e.syncWriteViews();
                    if (A("DEBUG", !1)) {
                        let r = []
                          , n = 0;
                        for (let i of t.programs) {
                            let t = performance.now();
                            this.handler.bindFrameBuffer(i.frameBuffer, i.width, i.height);
                            for (let {buffer: e, uniformIndex: t} of i.inputs)
                                await e.bindToReadTexture(t);
                            i.output.bindToDrawTexture(),
                            this.handler.useProgram(i.program);
                            for (let t of i.uniforms)
                                t.func.apply(e, t.args);
                            e.vertexAttribPointer(i.xyAttribLoc, 2, e.FLOAT, !0, 8, 0),
                            e.enableVertexAttribArray(i.xyAttribLoc),
                            e.drawArrays(e.TRIANGLE_STRIP, 0, Y.length / 2),
                            await this.handler.waitForComplete();
                            let a = performance.now() - t;
                            n += a;
                            let o = [];
                            for (let {buffer: e} of i.inputs)
                                e.unbindFromReadTexture(),
                                await e.syncReadViews(),
                                o.push(e.array.slice());
                            i.output.unbindFromDrawTexture(),
                            await i.output.syncReadViews();
                            let s = i.output.array.slice();
                            r.push({
                                Kernel: i.name,
                                "Elapsed time [ms]": a,
                                xs: o,
                                y: s
                            })
                        }
                        let i = Array.from(Object.values(r.reduce((e,t)=>(t.Kernel in e || (e[t.Kernel] = {
                            Kernel: t.Kernel,
                            Count: 0,
                            "Elapsed time [ms]": 0
                        }),
                        e[t.Kernel].Count++,
                        e[t.Kernel]["Elapsed time [ms]"] += t["Elapsed time [ms]"],
                        e), {})));
                        i.forEach(e=>e["Ratio [%]"] = (e["Elapsed time [ms]"] / n).toFixed(2)),
                        console.table(r),
                        console.table(i)
                    } else
                        for (let r of t.programs) {
                            this.handler.bindFrameBuffer(r.frameBuffer, r.width, r.height);
                            for (let {buffer: e, uniformIndex: t} of r.inputs)
                                await e.bindToReadTexture(t);
                            r.output.bindToDrawTexture(),
                            this.handler.useProgram(r.program);
                            for (let t of r.uniforms)
                                t.func.apply(e, t.args);
                            e.vertexAttribPointer(r.xyAttribLoc, 2, e.FLOAT, !0, 8, 0),
                            e.enableVertexAttribArray(r.xyAttribLoc),
                            e.drawArrays(e.TRIANGLE_STRIP, 0, Y.length / 2);
                            for (let {buffer: e} of r.inputs)
                                e.unbindFromReadTexture();
                            r.output.unbindFromDrawTexture()
                        }
                    for (let e of t.outputs)
                        await e.syncReadViews()
                }
            }
        }
        let K;
        class q {
            constructor() {
                if (this.pipelineStates = new Map,
                !J)
                    throw new Error("This browser does not support WebMetal");
                let e;
                try {
                    e = Q ? document.createElement("canvas").getContext("webgpu") : document.createElement("canvas").getContext("webmetal")
                } catch (e) {
                    throw new Error(`During initializing WebMetalRenderingContext, unexpected error is occurred: ${e.message}`)
                }
                if (!e)
                    throw new Error("WebMetalRenderingContext initialization failed");
                this.context = e,
                this.commandQueue = e.createCommandQueue(),
                this.loadKernel("kernel void sync(){}", "basic")
            }
            static getInstance() {
                return K || (K = new q),
                K
            }
            createBuffer(e) {
                return this.context.createBuffer(e)
            }
            loadKernel(e, t="") {
                let r = this.context.createLibrary(e);
                for (let e of r.functionNames) {
                    let n = r.functionWithName(e)
                      , i = this.context.createComputePipelineState(n);
                    this.pipelineStates.set(t + "." + e, i)
                }
            }
            createCommandBuffer() {
                return this.commandQueue.createCommandBuffer()
            }
            getPipelineStateByName(e) {
                let t = this.pipelineStates.get(e);
                if (!t)
                    throw TypeError(`Kernel function "${e}" is not loaded.`);
                return t
            }
            executeSinglePipelineState(e, t, r, n, i) {
                let a = this.commandBuffer || (this.commandBuffer = this.createCommandBuffer())
                  , o = a.createComputeCommandEncoder();
                o.setComputePipelineState(this.getPipelineStateByName(e));
                for (let e = 0; e < n.length; e++) {
                    let t, r = n[e];
                    t = r instanceof ee ? r.buffer : r,
                    o.setBuffer(t, 0, e)
                }
                o.dispatch(t, r),
                o.endEncoding();
                let s = null;
                return i && (s = a.completed),
                this.commandBuffer = null,
                a.commit(),
                s
            }
            async sync() {
                let e = this.createCommandBuffer()
                  , t = e.createComputeCommandEncoder();
                t.setComputePipelineState(this.getPipelineStateByName("basic.sync")),
                t.dispatch({
                    width: 1,
                    height: 1,
                    depth: 1
                }, {
                    width: 1,
                    height: 1,
                    depth: 1
                }),
                t.endEncoding();
                let r = e.completed;
                return e.commit(),
                r
            }
        }
        const Q = "WebGPURenderingContext"in window && "WebGPUComputeCommandEncoder"in window
          , J = "WebMetalRenderingContext"in window && "WebMetalComputeCommandEncoder"in window || Q;
        class ee extends G {
            constructor(e) {
                super(e, "webgpu"),
                0 == e && (e = 4),
                this.handler = q.getInstance(),
                this.buffer = this.handler.createBuffer(new Uint8Array(e)),
                this.bufferView = new Uint8Array(this.buffer.contents)
            }
            async write(e, t) {
                await this.handler.sync(),
                new e.constructor(this.bufferView.buffer).set(e, t)
            }
            async read(e, t=0, r) {
                if (!e)
                    throw new Error("dst cannot be null");
                if (await this.handler.sync(),
                0 !== this.byteLength) {
                    let n = e.constructor
                      , i = new n(this.bufferView.buffer,this.bufferView.byteOffset + t * n.BYTES_PER_ELEMENT,r);
                    e.set(i)
                }
            }
            getWriteView(e, t, r) {
                return new r(this.bufferView.buffer,this.bufferView.byteOffset + e * r.BYTES_PER_ELEMENT,t)
            }
            getReadView(e, t, r) {
                return new r(this.bufferView.buffer,this.bufferView.byteOffset + e * r.BYTES_PER_ELEMENT,t)
            }
            async syncWriteViews() {}
            async syncReadViews() {
                await this.handler.sync()
            }
        }
        const te = navigator.userAgent.includes("iPhone") || navigator.userAgent.includes("iPad");
        var re, ne;
        !function(e) {
            e[e.CHW = 0] = "CHW",
            e[e.HWC = 1] = "HWC"
        }(re || (re = {})),
        function(e) {
            e[e.RGB = 0] = "RGB",
            e[e.BGR = 1] = "BGR",
            e[e.GREY = 2] = "GREY",
            e[e.RGBA = 3] = "RGBA",
            e[e.BGRA = 4] = "BGRA"
        }(ne || (ne = {})),
        r.d(t, "getConfiguration", function() {
            return A
        }),
        r.d(t, "setConfiguration", function() {
            return S
        }),
        r.d(t, "getBackendAvailability", function() {
            return T
        }),
        r.d(t, "load", function() {
            return C
        }),
        r.d(t, "Math", function() {
            return R
        }),
        r.d(t, "Image", function() {
            return B
        });
        let ie = {};
        const ae = {
            webgpu: class extends j {
                constructor(e={}) {
                    super(e),
                    this.backendName = "webgpu"
                }
                static checkAvailability() {
                    return J
                }
                async init() {
                    this.webmetalHandler = q.getInstance(),
                    await this.checkIncompatibleGPU()
                }
                async checkIncompatibleGPU() {
                    this.webmetalHandler.loadKernel("\n#include <metal_stdlib>\nusing namespace metal;\n        kernel void check_compatibility(\n            device uint *A[[buffer(0)]],\n            uint global_index[[thread_position_in_grid]],\n            uint thread_execution_width[[thread_execution_width]]\n        ){\n            if (global_index == 0) {\n                A[0] = thread_execution_width;\n            }\n        }", "basic");
                    let e = this.webmetalHandler.createBuffer(new Uint32Array(1));
                    await this.webmetalHandler.executeSinglePipelineState("basic.check_compatibility", {
                        width: 1,
                        height: 1,
                        depth: 1
                    }, {
                        width: 1,
                        height: 1,
                        depth: 1
                    }, [e], !0);
                    let t = new Uint32Array(e.contents)[0];
                    if (32 != t)
                        throw new Error(`Sorry, this GPU does not compatible with WebMetal (thread_execution_width == ${t}. See checkIncompatibleGPU method of https://github.com/mil-tokyo/webdnn/blob/master/src/descriptor_runner/descriptor_runner/descriptor_runner_webmetal.ts`)
                }
                async fetchDescriptor(e) {
                    return (await i(`${e}/graph_${this.backendName}.json`, this.transformUrlDelegate)).json()
                }
                async fetchParameters(e, t) {
                    return a(await i(`${e}/weight_${this.backendName}.bin`, this.transformUrlDelegate), t)
                }
                async restoreCachedDescriptor(e) {
                    return D.getItem(`${e}_${this.backendName}_descriptor`).catch(()=>null)
                }
                async restoreCachedParameters(e, t) {
                    let r = await D.getItem(`${e}_${this.backendName}_parameters`).catch(()=>null);
                    return r && t && t(r.byteLength, r.byteLength),
                    r
                }
                async saveCache(e, t, r) {
                    await Promise.all([D.setItem(`${e}_${this.backendName}_descriptor`, t), D.setItem(`${e}_${this.backendName}_parameters`, r)])
                }
                async setDescriptorAndParameters(e, t) {
                    this.descriptor = e,
                    this.staticBuffer = null,
                    this.dynamicBuffer = null,
                    this.metaBuffers = null,
                    this.placeholderContext = new L.a(e.placeholders),
                    this.executionInfos = e.exec_infos,
                    this.webmetalHandler.loadKernel(this.descriptor.kernel_source, "descriptor"),
                    await this.initializeStaticBuffer(t),
                    await this.initializeMetaBuffers(),
                    await this.setPlaceholderValue({
                        __MAX_THREADS_PER_THREADGROUP__: te ? 512 : 1024
                    }),
                    this.placeholderContext && this.placeholderContext.isResolved && await this.initializeDynamicBuffer()
                }
                async initializeStaticBuffer(e) {
                    if (!this.descriptor)
                        throw Error("GraphDescriptor is not loaded.");
                    let t = this.descriptor
                      , r = new ee(t.memory_layout.static.size * Float32Array.BYTES_PER_ELEMENT);
                    this.staticBuffer = r;
                    let i = n(t.weight_encoding);
                    await r.write(await i.decode(new Uint8Array(e))),
                    (await this.getInputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                        e.buffer = r.bufferView.buffer
                    }
                    ),
                    (await this.getOutputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                        e.buffer = r.bufferView.buffer
                    }
                    )
                }
                async initializeMetaBuffers() {
                    if (!this.descriptor)
                        throw Error("GraphDescriptor is not loaded.");
                    this.metaBuffers = await Promise.all(this.descriptor.exec_infos.map(async e=>{
                        let t = new ee(e.meta_buffer.length * Int32Array.BYTES_PER_ELEMENT);
                        return await t.write(new Uint8Array(e.meta_buffer)),
                        t
                    }
                    ))
                }
                async initializeDynamicBuffer() {
                    if (!this.descriptor)
                        throw Error("GraphDescriptor is not loaded.");
                    if (!this.placeholderContext)
                        throw Error("PlaceholderContext is not initialized.");
                    let e = this.descriptor
                      , t = this.placeholderContext.resolve(e.memory_layout.dynamic.size)
                      , r = new ee(t * Float32Array.BYTES_PER_ELEMENT);
                    this.dynamicBuffer = r,
                    (await this.getInputViews()).filter(e=>e.isDynamic).forEach(e=>{
                        e.buffer = r.bufferView.buffer
                    }
                    ),
                    (await this.getOutputViews()).filter(e=>e.isDynamic).forEach(e=>{
                        e.buffer = r.bufferView.buffer
                    }
                    )
                }
                async setPlaceholderValue(e) {
                    if (!this.placeholderContext)
                        throw new Error("PlaceholderContext is not initialized.");
                    let t = this.placeholderContext;
                    if (t.update(e),
                    !t.isResolved)
                        return;
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (!this.metaBuffers)
                        throw new Error("MetaBuffers are not initialized");
                    let r = this.descriptor
                      , n = this.metaBuffers;
                    await this.initializeDynamicBuffer(),
                    this.executionInfos = await Promise.all(r.exec_infos.map(async(e,r)=>{
                        let i = new Int32Array(n[r].bufferView.buffer);
                        for (let r of e.unresolved_value_list)
                            i[r.offset] = t.resolve(r.placeholder);
                        return t.resolve(e)
                    }
                    ))
                }
                getInputViews() {
                    if (this.inputs)
                        return this.inputs;
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (!this.placeholderContext)
                        throw new Error("PlaceholderContext is not initialized");
                    let e = this.descriptor
                      , t = this.placeholderContext;
                    return this.inputs = e.inputs.map(r=>{
                        let n = e.memory_layout.static.allocations[r] || e.memory_layout.dynamic.allocations[r];
                        return new M(null,n.offset * M.BYTES_PER_ELEMENT,n.size,t)
                    }
                    ),
                    this.inputs
                }
                getOutputViews() {
                    if (this.outputs)
                        return this.outputs;
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (!this.placeholderContext)
                        throw new Error("PlaceholderContext is not initialized");
                    let e = this.descriptor
                      , t = this.placeholderContext;
                    return this.outputs = e.outputs.map(r=>{
                        let n = e.memory_layout.static.allocations[r] || e.memory_layout.dynamic.allocations[r];
                        return new M(null,n.offset * M.BYTES_PER_ELEMENT,n.size,t)
                    }
                    ),
                    this.outputs
                }
                async run() {
                    if (!this.executionInfos)
                        throw new Error("ExecutionInfos is not loaded");
                    if (!this.staticBuffer)
                        throw new Error("StaticBuffer is not initialized");
                    if (!this.dynamicBuffer)
                        throw new Error("DynamicBuffer is not initialized");
                    if (!this.metaBuffers)
                        throw new Error("MetaBuffer is not initialized");
                    if (!this.placeholderContext)
                        throw new Error("PlaceholderContext is not initialized");
                    if (!this.placeholderContext.isResolved)
                        throw new Error(`Not all placeholders are resolved: ${this.placeholderContext}`);
                    let e = this.staticBuffer
                      , t = this.dynamicBuffer
                      , r = this.metaBuffers;
                    if (!A("DEBUG", !1)) {
                        let n = null;
                        for (let i = 0; i < this.executionInfos.length; i++) {
                            let a = this.executionInfos[i]
                              , o = i == this.executionInfos.length - 1;
                            n = this.webmetalHandler.executeSinglePipelineState("descriptor." + a.entry_func_name, a.threadgroups_per_grid, a.threads_per_thread_group, [e, t, r[i]], o)
                        }
                        return n
                    }
                    {
                        let n = []
                          , i = 0;
                        for (let a = 0; a < this.executionInfos.length; a++) {
                            let o = this.executionInfos[a]
                              , s = performance.now();
                            await this.webmetalHandler.executeSinglePipelineState("descriptor." + o.entry_func_name, o.threadgroups_per_grid, o.threads_per_thread_group, [e, t, r[a]], !0);
                            let l = performance.now() - s;
                            n.push({
                                Kernel: o.entry_func_name,
                                "Elapsed time [ms]": l
                            }),
                            i += l
                        }
                        let a = Array.from(Object.values(n.reduce((e,t)=>(t.Kernel in e || (e[t.Kernel] = {
                            Kernel: t.Kernel,
                            Count: 0,
                            "Elapsed time [ms]": 0
                        }),
                        e[t.Kernel].Count++,
                        e[t.Kernel]["Elapsed time [ms]"] += t["Elapsed time [ms]"],
                        e), {})));
                        a.forEach(e=>e["Ratio [%]"] = (e["Elapsed time [ms]"] / i).toFixed(2)),
                        console.table(n),
                        console.table(a)
                    }
                }
            }
            ,
            webgl: Z,
            webassembly: W,
            fallback: class extends j {
                constructor(e={}) {
                    super(e),
                    this.backendName = "fallback"
                }
                static checkAvailability() {
                    return !0
                }
                async init() {}
                async setDescriptorAndParameters(e, t) {
                    this.setDescriptor(e),
                    await this.compile(),
                    await this.initializeStaticBuffer(t),
                    this.placeholderContext && this.placeholderContext.isResolved && await this.initializeDynamicBuffer()
                }
                async fetchDescriptor(e) {
                    return this.directory = e,
                    (await i(`${e}/graph_${this.backendName}.json`, this.transformUrlDelegate)).json()
                }
                async fetchParameters(e, t) {
                    return a(await i(`${e}/weight_${this.backendName}.bin`, this.transformUrlDelegate), t)
                }
                async restoreCachedDescriptor(e) {
                    return D.getItem(`${e}_${this.backendName}_descriptor`).catch(()=>null)
                }
                async restoreCachedParameters(e, t) {
                    let r = await D.getItem(`${e}_${this.backendName}_parameters`).catch(()=>null);
                    return r && t && t(r.byteLength, r.byteLength),
                    r
                }
                async saveCache(e, t, r) {
                    await Promise.all([D.setItem(`${e}_${this.backendName}_descriptor`, t), D.setItem(`${e}_${this.backendName}_parameters`, r)])
                }
                setDescriptor(e) {
                    this.descriptor = e,
                    this.placeholderContext = new L.a,
                    this.placeholderContext.update(e.placeholders),
                    this.kernelObj = null,
                    this.variableMap = null,
                    this.staticBuffer = null,
                    this.dynamicBuffer = null
                }
                async compile() {
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    await new Promise(e=>{
                        let t = document.createElement("script");
                        t.type = "text/javascript",
                        t.readyState ? t.onreadystatechange = (()=>{
                            ("loaded" == t.readyState || "complete" == t.readyState) && (t.onreadystatechange = null,
                            e())
                        }
                        ) : t.onload = e,
                        t.src = this.transformUrlDelegate(`${this.directory}/kernels_fallback.js`),
                        document.getElementsByTagName("head")[0].appendChild(t)
                    }
                    ),
                    this.kernelObj = window.dnn_fallback_kernel
                }
                async initializeStaticBuffer(e) {
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    let t = this.descriptor
                      , r = new Float32Array(t.memory_layout.static.size);
                    this.staticBuffer = r;
                    let i = this.variableMap || new Map;
                    this.variableMap = i,
                    Object.entries(t.memory_layout.static.allocations).forEach(([e,t])=>{
                        i.set(e, new Float32Array(r.buffer,t.offset * Float32Array.BYTES_PER_ELEMENT,t.size))
                    }
                    );
                    let a = n(this.descriptor.weight_encoding);
                    r.set(await a.decode(new Uint8Array(e))),
                    (await this.getInputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                        e.buffer = r.buffer
                    }
                    ),
                    (await this.getOutputViews()).filter(e=>!e.isDynamic).forEach(e=>{
                        e.buffer = r.buffer
                    }
                    )
                }
                async initializeDynamicBuffer() {
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (!this.placeholderContext)
                        throw new Error("PlaceholderContext is not initialized");
                    let e = this.descriptor
                      , t = this.placeholderContext
                      , r = new Float32Array(t.resolve(e.memory_layout.dynamic.size));
                    this.dynamicBuffer = r;
                    let n = this.variableMap || new Map;
                    this.variableMap = n,
                    Object.entries(e.memory_layout.dynamic.allocations).forEach(([e,i])=>{
                        n.set(e, new Float32Array(r.buffer,t.resolve(i.offset) * Float32Array.BYTES_PER_ELEMENT,t.resolve(i.size)))
                    }
                    ),
                    (await this.getInputViews()).filter(e=>e.isDynamic).forEach(e=>{
                        e.buffer = r.buffer
                    }
                    ),
                    (await this.getOutputViews()).filter(e=>e.isDynamic).forEach(e=>{
                        e.buffer = r.buffer
                    }
                    )
                }
                async setPlaceholderValue(e) {
                    if (!this.placeholderContext)
                        throw new Error("placeholderContext is not initialized");
                    let t = this.placeholderContext;
                    t.update(e),
                    t.isResolved && await this.initializeDynamicBuffer()
                }
                async run() {
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (!this.placeholderContext)
                        throw new Error("placeholderContext is not initialized");
                    if (!this.variableMap)
                        throw new Error("Variable map is not initialized");
                    if (!this.staticBuffer)
                        throw new Error("StaticBuffer map is not initialized");
                    if (!this.dynamicBuffer)
                        throw new Error("DynamicBuffer map is not initialized");
                    let e = this.variableMap
                      , t = this.placeholderContext
                      , r = this.descriptor.exec_infos.map(e=>t.resolve(e))
                      , n = Date.now()
                      , i = Date.now();
                    for (let t, a = 0; a < r.length; a++) {
                        1e3 <= (t = Date.now()) - i && (console.log(`Processed ${a}/${r.length} kernels in ${t - n} ms`),
                        i = t,
                        await o());
                        let s = r[a]
                          , l = s.inputs.map(t=>e.get(t))
                          , c = s.outputs.map(t=>e.get(t));
                        this.kernelObj[s.entry_func_name](l, c, s.call_option)
                    }
                    console.log(`Processed ${r.length}/${r.length} kernels in ${Date.now() - n} ms`)
                }
                getInputViews() {
                    if (this.inputs)
                        return this.inputs;
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (!this.placeholderContext)
                        throw new Error("PlaceholderContext is not initialized");
                    let e = this.descriptor
                      , t = this.placeholderContext;
                    return this.inputs = e.inputs.map(r=>{
                        let n = e.memory_layout.static.allocations[r] || e.memory_layout.dynamic.allocations[r];
                        return new M(null,n.offset * M.BYTES_PER_ELEMENT,n.size,t)
                    }
                    ),
                    this.inputs
                }
                getOutputViews() {
                    if (this.outputs)
                        return this.outputs;
                    if (!this.descriptor)
                        throw new Error("Descriptor is not loaded");
                    if (!this.placeholderContext)
                        throw new Error("PlaceholderContext is not initialized");
                    let e = this.descriptor
                      , t = this.placeholderContext;
                    return this.outputs = e.outputs.map(r=>{
                        let n = e.memory_layout.static.allocations[r] || e.memory_layout.dynamic.allocations[r];
                        return new M(null,n.offset * M.BYTES_PER_ELEMENT,n.size,t)
                    }
                    ),
                    this.outputs
                }
            }
        }
    }
    ])
});
