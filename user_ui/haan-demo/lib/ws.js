"use strict";
/**
 * 무의존성(zero-dependency) 최소 WebSocket 서버 (RFC 6455 텍스트 프레임).
 *
 * 데모용: `npm install` 없이 `node server.js`만으로 동작하도록 표준 라이브러리(crypto)만
 * 사용해 핸드셰이크 + 텍스트 프레임 송수신을 구현한다.
 *
 * ⚠️ 실서비스 전환 시: 검증된 라이브러리(`ws`)와 wss(TLS), 백프레셔/압축 처리로 교체할 것.
 * 여기서는 작은 JSON 메시지 브로드캐스트만 처리하면 충분하다.
 */
const crypto = require("crypto");
const { EventEmitter } = require("events");

const GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/** 단일 소켓 연결을 감싸는 얇은 래퍼. */
class WSConn extends EventEmitter {
  constructor(socket) {
    super();
    this.socket = socket;
    this.alive = true;
    this._buf = Buffer.alloc(0);
    this.meta = {}; // role 등 임의 메타데이터 저장용

    socket.on("data", (chunk) => this._onData(chunk));
    socket.on("close", () => this._die());
    socket.on("error", () => this._die());
  }

  _die() {
    if (!this.alive) return;
    this.alive = false;
    this.emit("close");
  }

  _onData(chunk) {
    this._buf = Buffer.concat([this._buf, chunk]);
    // 버퍼에 완전한 프레임이 여러 개 쌓일 수 있으므로 반복 파싱
    while (true) {
      const frame = this._readFrame();
      if (!frame) break;
      const { opcode, payload } = frame;
      if (opcode === 0x8) {
        // close
        this.close();
        break;
      } else if (opcode === 0x9) {
        // ping -> pong
        this._send(0xa, payload);
      } else if (opcode === 0xa) {
        // pong: no-op
      } else if (opcode === 0x1) {
        // text
        this.emit("message", payload.toString("utf8"));
      }
      // 0x2(binary), 연속 프레임 등은 데모에서 사용하지 않음
    }
  }

  /** 버퍼에서 완전한 한 프레임을 떼어낸다. 부족하면 null. */
  _readFrame() {
    const buf = this._buf;
    if (buf.length < 2) return null;
    const b0 = buf[0];
    const b1 = buf[1];
    const opcode = b0 & 0x0f;
    const masked = (b1 & 0x80) !== 0;
    let len = b1 & 0x7f;
    let offset = 2;

    if (len === 126) {
      if (buf.length < offset + 2) return null;
      len = buf.readUInt16BE(offset);
      offset += 2;
    } else if (len === 127) {
      if (buf.length < offset + 8) return null;
      // 데모 메시지는 4GB를 넘지 않으므로 하위 32비트만 사용
      const hi = buf.readUInt32BE(offset);
      const lo = buf.readUInt32BE(offset + 4);
      len = hi * 2 ** 32 + lo;
      offset += 8;
    }

    let mask = null;
    if (masked) {
      if (buf.length < offset + 4) return null;
      mask = buf.slice(offset, offset + 4);
      offset += 4;
    }

    if (buf.length < offset + len) return null; // 페이로드가 아직 다 안 옴

    let payload = buf.slice(offset, offset + len);
    if (masked && mask) {
      const out = Buffer.allocUnsafe(len);
      for (let i = 0; i < len; i++) out[i] = payload[i] ^ mask[i & 3];
      payload = out;
    }

    this._buf = buf.slice(offset + len); // 소비한 만큼 제거
    return { opcode, payload };
  }

  /** 서버->클라이언트 프레임 인코딩 (마스킹 없음). */
  _send(opcode, payload) {
    if (!this.alive) return;
    const len = payload.length;
    let header;
    if (len < 126) {
      header = Buffer.alloc(2);
      header[1] = len;
    } else if (len < 65536) {
      header = Buffer.alloc(4);
      header[1] = 126;
      header.writeUInt16BE(len, 2);
    } else {
      header = Buffer.alloc(10);
      header[1] = 127;
      header.writeUInt32BE(Math.floor(len / 2 ** 32), 2);
      header.writeUInt32BE(len >>> 0, 6);
    }
    header[0] = 0x80 | opcode; // FIN + opcode
    try {
      this.socket.write(Buffer.concat([header, payload]));
    } catch {
      this._die();
    }
  }

  /** JSON 객체 또는 문자열 전송. */
  send(data) {
    const text = typeof data === "string" ? data : JSON.stringify(data);
    this._send(0x1, Buffer.from(text, "utf8"));
  }

  close() {
    if (!this.alive) return;
    try {
      this._send(0x8, Buffer.alloc(0));
      this.socket.end();
    } catch {
      /* noop */
    }
    this._die();
  }
}

/**
 * 기존 http.Server에 WebSocket 업그레이드를 부착한다.
 * @returns {EventEmitter} 'connection' 이벤트로 WSConn 을 넘겨줌.
 */
function attachWebSocket(httpServer) {
  const hub = new EventEmitter();
  httpServer.on("upgrade", (req, socket) => {
    const key = req.headers["sec-websocket-key"];
    if (!key) {
      socket.destroy();
      return;
    }
    const accept = crypto
      .createHash("sha1")
      .update(key + GUID)
      .digest("base64");
    const headers = [
      "HTTP/1.1 101 Switching Protocols",
      "Upgrade: websocket",
      "Connection: Upgrade",
      `Sec-WebSocket-Accept: ${accept}`,
      "\r\n",
    ].join("\r\n");
    socket.write(headers);
    const conn = new WSConn(socket);
    hub.emit("connection", conn, req);
  });
  return hub;
}

module.exports = { attachWebSocket, WSConn };
