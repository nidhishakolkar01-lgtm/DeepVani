const { app, BrowserWindow, systemPreferences, session } = require('electron');
const path = require('path');
const http = require('http');
const fs = require('fs');

// Append Chromium command-line switches for instant WebRTC camera stream allocation
app.commandLine.appendSwitch('use-fake-ui-for-media-stream');
app.commandLine.appendSwitch('enable-usermedia-screen-capturing');
app.commandLine.appendSwitch('allow-file-access-from-files');
app.commandLine.appendSwitch('ignore-certificate-errors');

let mainWindow;
let localServer;
let activePort = 8000;

function startLocalServer(callback) {
  const rootDir = __dirname;
  
  const createServerInstance = (port) => {
    const server = http.createServer((req, res) => {
      let filePath = path.join(rootDir, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
      
      if (!filePath.startsWith(rootDir)) {
        res.writeHead(403);
        return res.end('Access Denied');
      }

      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes = {
        '.html': 'text/html',
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.ico': 'image/x-icon'
      };

      const contentType = mimeTypes[ext] || 'application/octet-stream';

      fs.readFile(filePath, (err, content) => {
        if (err) {
          if (err.code === 'ENOENT') {
            res.writeHead(404);
            res.end('404 Not Found');
          } else {
            res.writeHead(500);
            res.end('500 Server Error');
          }
        } else {
          res.writeHead(200, {
            'Content-Type': contentType,
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'no-cache'
          });
          res.end(content, 'utf-8');
        }
      });
    });

    server.on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        console.log(`Port ${port} in use. Falling back to port ${port + 1}...`);
        if (port < 8010) {
          createServerInstance(port + 1);
        } else {
          activePort = 8000;
          if (callback) callback();
        }
      } else {
        console.error('Server error:', err);
        if (callback) callback();
      }
    });

    server.listen(port, '127.0.0.1', () => {
      activePort = port;
      localServer = server;
      console.log(`DeepVani Server active on http://127.0.0.1:${activePort}`);
      if (callback) callback();
    });
  };

  createServerInstance(activePort);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 850,
    minWidth: 900,
    minHeight: 650,
    title: 'DeepVani — Silent Speech Recognition Desktop',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false
    },
    autoHideMenuBar: true,
    backgroundColor: '#020617'
  });

  mainWindow.loadURL(`http://127.0.0.1:${activePort}`);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    callback(true);
  });

  session.defaultSession.setPermissionCheckHandler(() => {
    return true;
  });

  if (systemPreferences && systemPreferences.askForMediaAccess) {
    try {
      await systemPreferences.askForMediaAccess('camera');
    } catch (e) {}
  }

  startLocalServer(() => {
    createWindow();
  });
});

app.on('window-all-closed', () => {
  if (localServer) localServer.close();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
