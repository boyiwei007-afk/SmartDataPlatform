/**
 * SmartAnalysis by WBY — Frontend Application Logic
 *
 * Tab structure:
 *   Tab 1 "import"       — Data upload (mostly unchanged)
 *   Tab 2 "understanding" — Stepper: Data Overview + Data Cleaning
 *   Tab 3 "analysis"      — AI-assisted analysis notebook (SSE streaming)
 *   Tab 4 "training"      — Model training (unchanged)
 *   Tab 5 "sandbox"       — What-If sandbox (unchanged)
 */
(function () {
  "use strict";

  var $ = function (sel) { return document.querySelector(sel); };
  var $$ = function (sel) { return document.querySelectorAll(sel); };

  // =======================================================================
  // Global State
  // =======================================================================
  var currentSchema = null;
  var currentFilename = null;        // active working filename (may be cleaned)
  var originalFilename = null;       // original uploaded filename
  var originalSchema = null;         // schema before cleaning
  var profileData = null;            // cached /data-profile result
  var activeTab = "import";
  var activeStep = "overview";       // Tab 2 stepper state
  var analysisHistory = [];          // Tab 3 analysis sessions
  var activeAnalysisId = null;
  var analysisChartInstances = {};   // ECharts instances per analysis step
  var analysisLoading = false;

  var STORAGE_KEYS = {
    apiKey: "smartanalysis_api_key",
    baseUrl: "smartanalysis_base_url",
    modelName: "smartanalysis_model_name",
  };
  var ALLOWED_EXTENSIONS = [".csv", ".xls", ".xlsx"];
  var MAX_FILE_SIZE = 50 * 1024 * 1024;
  var API_BASE = "http://localhost:8000";

  var TAB_ORDER = ["import", "understanding", "analysis", "training", "sandbox", "admin"];

  // =======================================================================
  // Error Modal (unified entry point for all user-facing errors)
  // =======================================================================
  function showErrorModal(title, message, hint) {
    // Remove any existing modal
    var existing = document.querySelector(".error-modal-overlay");
    if (existing) existing.remove();

    var overlay = document.createElement("div");
    overlay.className = "error-modal-overlay";

    var card = document.createElement("div");
    card.className = "error-modal-card";

    var iconWrap = document.createElement("div");
    iconWrap.className = "error-modal-icon error";
    iconWrap.innerHTML = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="13"/><circle cx="12" cy="16.5" r="0.8" fill="currentColor" stroke="none"/></svg>';

    var titleEl = document.createElement("div");
    titleEl.className = "error-modal-title";
    titleEl.textContent = title;

    var msgEl = document.createElement("div");
    msgEl.className = "error-modal-message";
    msgEl.textContent = message;

    card.appendChild(iconWrap);
    card.appendChild(titleEl);
    card.appendChild(msgEl);

    if (hint) {
      var hintEl = document.createElement("div");
      hintEl.className = "error-modal-hint";
      hintEl.textContent = hint;
      card.appendChild(hintEl);
    }

    var actions = document.createElement("div");
    actions.className = "error-modal-actions";
    var closeBtn = document.createElement("button");
    closeBtn.className = "error-modal-btn primary";
    closeBtn.textContent = "知道了";
    closeBtn.addEventListener("click", function () { overlay.remove(); });
    actions.appendChild(closeBtn);
    card.appendChild(actions);

    overlay.appendChild(card);
    // Click overlay to dismiss
    overlay.addEventListener("click", function (e) { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
  }

  // =======================================================================
  // Global LLM Settings
  // =======================================================================
  function getLlmConfig() {
    return {
      api_key: sessionStorage.getItem(STORAGE_KEYS.apiKey) || "",
      base_url: sessionStorage.getItem(STORAGE_KEYS.baseUrl) || "",
      model_name: sessionStorage.getItem(STORAGE_KEYS.modelName) || "",
    };
  }

  function isLlmConfigured() {
    var c = getLlmConfig();
    return !!(c.api_key && c.base_url && c.model_name);
  }

  function requireLlmOrPrompt(callback) {
    if (isLlmConfigured()) { callback(); return; }
    showToast("请先在左侧「LLM 配置」中填写 API Key、Base URL 和 Model Name");
  }

  // =======================================================================
  // Utility
  // =======================================================================
  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function showToast(msg, dur) {
    if (!dur) dur = 4000;
    var c = $("#toastContainer"); if (!c) return;
    var t = document.createElement("div"); t.className = "toast";
    t.innerHTML = '<svg class="toast-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01M12 3l9.66 16.5H2.34L12 3z"/></svg><span>' + escapeHtml(msg) + '</span>';
    c.appendChild(t);
    var timer = setTimeout(function () { dismissToast(t); }, dur);
    t._timer = timer;
    t.addEventListener("click", function () { clearTimeout(t._timer); dismissToast(t); });
  }

  function dismissToast(t) {
    if (t._removing) return; t._removing = true;
    t.classList.add("removing");
    t.addEventListener("animationend", function () { if (t.parentNode) t.parentNode.removeChild(t); });
  }

  function dtypeTag(dtype) {
    var l = dtype, cls = "bg-gray-100 text-gray-500";
    switch (dtype) {
      case "numeric": l = "数值"; cls = "bg-blue-50 text-blue-600"; break;
      case "categorical": l = "分类"; cls = "bg-purple-50 text-purple-600"; break;
      case "datetime": l = "日期"; cls = "bg-cyan-50 text-cyan-600"; break;
      case "text": l = "文本"; cls = "bg-amber-50 text-amber-600"; break;
      case "boolean": l = "布尔"; cls = "bg-emerald-50 text-emerald-600"; break;
    }
    return '<span class="inline-block text-[10px] px-1.5 py-0.5 rounded-full ' + cls + '">' + l + '</span>';
  }

  // =======================================================================
  // Settings (inline in Tab 1 sidebar)
  // =======================================================================
  var globalApiKey = $("#globalApiKey"), globalBaseUrl = $("#globalBaseUrl");
  var globalModelName = $("#globalModelName"), settingsStatus = $("#settingsStatus");

  function loadSettingsToInputs() {
    globalApiKey.value = sessionStorage.getItem(STORAGE_KEYS.apiKey) || "";
    globalBaseUrl.value = sessionStorage.getItem(STORAGE_KEYS.baseUrl) || "";
    globalModelName.value = sessionStorage.getItem(STORAGE_KEYS.modelName) || "";
  }

  // =======================================================================
  // Auth — login / register / logout
  // =======================================================================
  var currentUser = null;  // {username, role, token}
  var AUTH_TOKEN_KEY = "smartanalysis_token";

  function loadAuthState() {
    var raw = sessionStorage.getItem(AUTH_TOKEN_KEY);
    if (raw) {
      try { currentUser = JSON.parse(raw); } catch (e) { currentUser = null; }
    }
    updateAuthUI();
  }

  function saveAuthState(user) {
    currentUser = user;
    if (user) {
      sessionStorage.setItem(AUTH_TOKEN_KEY, JSON.stringify(user));
    } else {
      sessionStorage.removeItem(AUTH_TOKEN_KEY);
    }
    updateAuthUI();
  }

  function updateAuthUI() {
    var form = $("#loginForm"), info = $("#loginInfo");
    var nameEl = $("#loginInfoName"), roleEl = $("#loginInfoRole");
    var mgmtBtn = $("#btnUserMgmt");
    var adminTab = $("#tabBtnAdmin");
    if (currentUser && currentUser.token) {
      if (form) form.classList.add("hidden");
      if (info) info.classList.remove("hidden");
      if (nameEl) nameEl.textContent = currentUser.username;
      if (roleEl) { roleEl.textContent = currentUser.role; roleEl.classList.remove("hidden"); }
      if (mgmtBtn) { if (currentUser.role === "admin") mgmtBtn.classList.remove("hidden"); else mgmtBtn.classList.add("hidden"); }
      if (adminTab) { if (currentUser.role === "admin") adminTab.classList.remove("hidden"); else adminTab.classList.add("hidden"); }
      loadUserHistory();
    } else {
      if (form) form.classList.remove("hidden");
      if (info) info.classList.add("hidden");
      if (adminTab) adminTab.classList.add("hidden");
      if (mgmtBtn) mgmtBtn.classList.add("hidden");
    }
  }

  function loadUserHistory() {
    if (!currentUser || !currentUser.token) return;
    // Load upload history
    _origFetch(API_BASE + "/history/uploads", {
      headers: { "Authorization": "Bearer " + currentUser.token },
    }).then(function (r) { return r.json(); })
      .then(function (d) { renderHistoryList(d.uploads || []); })
      .catch(function () { /* ignore */ });
  }

  function renderHistoryList(uploads) {
    var tl = $("#notebookTimeline");
    if (!tl || !currentUser) return;
    var html = "";
    if (uploads.length > 0) {
      html += '<p class="text-[10px] mb-1" style="color:#94A3B8;">上传记录</p>';
      uploads.slice(0, 10).forEach(function (u) {
        html += '<div class="notebook-timeline-item" style="font-size:0.6875rem; color:#64748B;">' +
          escapeHtml(u.original_name) + '<br/><span style="font-size:0.5625rem; color:#CBD5E1;">' + (u.uploaded_at || "").slice(0,16) + '</span></div>';
      });
    }
    if (analysisHistory.length > 0) {
      html += '<p class="text-[10px] mt-2 mb-1" style="color:#94A3B8;">分析记录</p>';
      analysisHistory.forEach(function (h) {
        html += '<div class="notebook-timeline-item" data-analysis-id="' + h.id + '" style="font-size:0.6875rem; color:#64748B;">' +
          escapeHtml(h.question.slice(0, 40)) + '<br/><span style="font-size:0.5625rem; color:#CBD5E1;">' + h.time + '</span></div>';
      });
    }
    if (!html) html = '<p class="text-xs" style="color:#94A3B8;">暂无历史记录</p>';
    tl.innerHTML = html;
  }

  function authFetch(url, options) {
    options = options || {};
    options.headers = options.headers || {};
    if (currentUser && currentUser.token) {
      options.headers["Authorization"] = "Bearer " + currentUser.token;
    }
    return fetch(url, options);
  }

  // Replace global fetch for all API calls — intercept and add auth header
  var _origFetch = window.fetch;
  window.fetch = function (url, options) {
    if (typeof url === "string" && url.indexOf(API_BASE) === 0 && url.indexOf("/login") === -1 && url.indexOf("/register") === -1) {
      options = options || {};
      options.headers = options.headers || {};
      if (currentUser && currentUser.token && !options.headers["Authorization"]) {
        options.headers["Authorization"] = "Bearer " + currentUser.token;
      }
    }
    return _origFetch(url, options).then(function (resp) {
      if (resp.status === 401) {
        showErrorModal("登录已过期", "您的登录状态已失效，请重新登录后再操作。");
        saveAuthState(null);
      }
      return resp;
    });
  };

  // Login
  $("#btnLogin").addEventListener("click", function () {
    var u = $("#loginUsername").value.trim();
    var p = $("#loginPassword").value.trim();
    if (!u || !p) { showLoginStatus("请填写用户名和密码", true); return; }
    _origFetch(API_BASE + "/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p }),
    }).then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        if (!res.ok) { showLoginStatus(res.data.detail || "登录失败", true); return; }
        saveAuthState({ username: res.data.username, role: res.data.role, token: res.data.token });
        showLoginStatus("已登录", false);
        $("#loginUsername").value = ""; $("#loginPassword").value = "";
        showPrivacyNotice();
        showHistoryPicker();
      }).catch(function () { showLoginStatus("网络错误", true); });
  });

  // Register
  $("#btnRegister").addEventListener("click", function () {
    var u = $("#loginUsername").value.trim();
    var p = $("#loginPassword").value.trim();
    if (!u || !p) { showLoginStatus("请填写用户名和密码", true); return; }
    if (p.length < 4) { showLoginStatus("密码至少 4 位", true); return; }
    _origFetch(API_BASE + "/register", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p }),
    }).then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        if (!res.ok) { showLoginStatus(res.data.detail || "注册失败", true); return; }
        showLoginStatus("注册成功，请登录", false);
      }).catch(function () { showLoginStatus("网络错误", true); });
  });

  // Logout
  $("#btnLogout").addEventListener("click", function () { saveAuthState(null); });

  // Self-delete account
  $("#btnDeleteAccount").addEventListener("click", function () {
    if (!currentUser || !currentUser.token) return;
    if (currentUser.username === "admin") { showToast("管理员账号不可注销"); return; }
    if (!confirm("确定注销账号「" + currentUser.username + "」吗？此操作不可撤销。")) return;
    _origFetch(API_BASE + "/users/" + currentUser.username, {
      method: "DELETE",
      headers: { "Authorization": "Bearer " + currentUser.token },
    }).then(function (r) { return r.json(); })
      .then(function (res) {
        if (res.ok) { showToast("账号已注销"); saveAuthState(null); }
        else { showToast(res.error || "注销失败"); }
      }).catch(function () { showToast("注销失败"); });
  });

  // User management (admin)
  $("#btnUserMgmt").addEventListener("click", function () { switchTab("admin"); });

  // Admin tab: load users + all history on switch
  function loadAdminUsers() {
    var token = currentUser ? currentUser.token : "";
    var headers = { "Authorization": "Bearer " + token };

    // Users
    _origFetch(API_BASE + "/users", { headers: headers })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var users = d.users || [];
        var h = '<table class="analysis-stats-table"><thead><tr><th>用户名</th><th>角色</th><th>注册时间</th><th>操作</th></tr></thead><tbody>';
        users.forEach(function (u) {
          h += '<tr><td class="font-medium">' + escapeHtml(u.username) + '</td><td>' + u.role + '</td><td style="font-size:0.6875rem;">' + (u.created_at || "").slice(0,10) + '</td>' +
            '<td>' + (u.username !== "admin" ? '<button class="text-xs text-red-500 hover:underline btn-del-user" data-username="' + escapeHtml(u.username) + '">删除</button>' : '—') + '</td></tr>';
        });
        h += '</tbody></table>';
        var ut = $("#adminUserTable"); if (ut) ut.innerHTML = h;
        // Bind delete
        $$(".btn-del-user").forEach(function (btn) {
          btn.addEventListener("click", function () {
            var uname = this.dataset.username;
            if (confirm("确定删除用户 " + uname + " 吗？")) {
              _origFetch(API_BASE + "/users/" + uname, { method: "DELETE", headers: headers })
                .then(function (r) { return r.json(); })
                .then(function (res) {
                  if (res.ok) { showToast("已删除 " + uname); loadAdminUsers(); }
                  else { showToast(res.error || "删除失败"); }
                }).catch(function () { showToast("删除失败"); });
            }
          });
        });
      }).catch(function () {});

    // Upload history (all users)
    _origFetch(API_BASE + "/admin/history/uploads", { headers: headers })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var rows = d.uploads || [];
        var h = '<table class="analysis-stats-table"><thead><tr><th>用户</th><th>文件名</th><th>时间</th><th>操作</th></tr></thead><tbody>';
        rows.slice(0, 30).forEach(function (r) {
          h += '<tr><td><span class="text-[10px] px-1 py-0.5 rounded" style="background:#F1F5F9;">' + escapeHtml(r.username) + '</span></td><td>' + escapeHtml(r.original_name) + '</td><td style="font-size:0.6875rem;">' + (r.uploaded_at || "").slice(0,16) + '</td>' +
            '<td><button class="text-xs text-red-400 hover:text-red-600 btn-del-upload" data-id="' + r.id + '">删除</button></td></tr>';
        });
        h += '</tbody></table>';
        if (rows.length === 0) h = '<p class="text-xs" style="color:#94A3B8;">暂无记录</p>';
        var el = $("#adminUploadsTable"); if (el) el.innerHTML = h;
        // Bind delete buttons
        $$(".btn-del-upload").forEach(function (btn) {
          btn.addEventListener("click", function () {
            var id = this.dataset.id;
            _origFetch(API_BASE + "/admin/history/uploads/" + id, { method: "DELETE", headers: headers })
              .then(function (r) { return r.json(); })
              .then(function () { loadAdminUsers(); })
              .catch(function () {});
          });
        });
      }).catch(function () {});

    // Analysis history (all users)
    _origFetch(API_BASE + "/admin/history/analysis", { headers: headers })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var rows = d.analysis || [];
        var h = '<table class="analysis-stats-table"><thead><tr><th>用户</th><th>提问</th><th>时间</th><th>操作</th></tr></thead><tbody>';
        rows.slice(0, 30).forEach(function (r) {
          h += '<tr><td><span class="text-[10px] px-1 py-0.5 rounded" style="background:#F1F5F9;">' + escapeHtml(r.username) + '</span></td><td>' + escapeHtml((r.question || "").slice(0, 60)) + '</td><td style="font-size:0.6875rem;">' + (r.created_at || "").slice(0,16) + '</td>' +
            '<td><button class="text-xs text-red-400 hover:text-red-600 btn-del-analysis" data-id="' + r.id + '">删除</button></td></tr>';
        });
        h += '</tbody></table>';
        if (rows.length === 0) h = '<p class="text-xs" style="color:#94A3B8;">暂无记录</p>';
        var el2 = $("#adminAnalysisTable"); if (el2) el2.innerHTML = h;
        $$(".btn-del-analysis").forEach(function (btn) {
          btn.addEventListener("click", function () {
            var id = this.dataset.id;
            _origFetch(API_BASE + "/admin/history/analysis/" + id, { method: "DELETE", headers: headers })
              .then(function (r) { return r.json(); })
              .then(function () { loadAdminUsers(); })
              .catch(function () {});
          });
        });
      }).catch(function () {});
  }

  // Hook admin tab switch
  var _origSwitchTab2 = switchTab;
  switchTab = function (tabName) {
    _origSwitchTab2(tabName);
    if (tabName === "admin") loadAdminUsers();
  };

  // Login status display
  $("#loginUsername").addEventListener("keydown", function (e) { if (e.key === "Enter") $("#btnLogin").click(); });
  $("#loginPassword").addEventListener("keydown", function (e) { if (e.key === "Enter") $("#btnLogin").click(); });

  function showLoginStatus(msg, isErr) {
    var el = $("#loginStatus");
    if (!el) return;
    el.textContent = msg;
    el.classList.remove("hidden", "text-green-600", "text-red-500");
    el.classList.add(isErr ? "text-red-500" : "text-green-600");
    setTimeout(function () { el.classList.add("hidden"); }, 3000);
  }

  // ---- Privacy notice ----
  function showPrivacyNotice() {
    if (localStorage.getItem("privacy_no_more")) return;
    $("#privacyOverlay").classList.remove("hidden");
  }

  $("#btnConfirmPrivacy").addEventListener("click", function () {
    var noMore = $("#chkNoMorePrivacy");
    if (noMore && noMore.checked) {
      localStorage.setItem("privacy_no_more", "1");
    }
    $("#privacyOverlay").classList.add("hidden");
  });

  // ---- History picker (after login) ----
  function showHistoryPicker() {
    if (!currentUser || !currentUser.token) return;
    _origFetch(API_BASE + "/history/uploads", {
      headers: { "Authorization": "Bearer " + currentUser.token },
    }).then(function (r) { return r.json(); })
      .then(function (d) {
        var uploads = d.uploads || [];
        var zone = $("#uploadZone");
        if (!zone) return;
        if (uploads.length === 0) return; // No history, keep default upload UI

        var html = '<p class="upload-title mb-3">欢迎回来，' + escapeHtml(currentUser.username) + '</p>';
        html += '<p class="upload-hint mb-4">选择继续之前的分析，或上传新数据</p>';
        html += '<div class="flex flex-col gap-2 mb-4" style="max-height:220px; overflow-y:auto; text-align:left;">';
        uploads.slice(0, 8).forEach(function (u) {
          html += '<div class="history-picker-item" data-filename="' + escapeHtml(u.filename) + '" data-name="' + escapeHtml(u.original_name) + '" style="display:flex; align-items:center; justify-content:space-between; padding:0.5rem 0.75rem; border:1px solid #E2E8F0; border-radius:0.5rem; cursor:pointer; transition:all 0.15s;">' +
            '<div><span class="text-sm font-medium" style="color:#1E293B;">' + escapeHtml(u.original_name) + '</span><br/><span class="text-[10px]" style="color:#94A3B8;">' + (u.uploaded_at || "").slice(0,16) + '</span></div>' +
            '<svg class="w-4 h-4" style="color:#94A3B8;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>' +
          '</div>';
        });
        html += '</div>';
        html += '<button id="btnNewUpload" class="text-sm font-medium px-4 py-2 rounded-lg border" style="color:#0052CC; border-color:#0052CC; background:#fff;">上传新文件</button>';
        zone.innerHTML = html;

        // Bind history item clicks
        zone.querySelectorAll(".history-picker-item").forEach(function (item) {
          item.addEventListener("click", function () {
            var fname = this.dataset.filename;
            resumeSession(fname);
          });
          item.addEventListener("mouseenter", function () { this.style.borderColor = "#0052CC"; this.style.background = "rgba(0,82,204,0.03)"; });
          item.addEventListener("mouseleave", function () { this.style.borderColor = "#E2E8F0"; this.style.background = ""; });
        });

        // "New upload" button
        var btnNew = $("#btnNewUpload");
        if (btnNew) {
          btnNew.addEventListener("click", function () {
            // Restore original upload zone
            zone.innerHTML = '<svg class="upload-icon" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg><p class="upload-title">点击或拖拽 CSV / Excel 文件至此处</p><p class="upload-hint">支持 .csv / .xls / .xlsx 格式，最大 50 MB</p>';
            zone.appendChild(fileInput);
          });
        }
      }).catch(function () {});
  }

  function resumeSession(filename) {
    // Just load the file schema — it's already on the server
    loadingIndicator.classList.remove("hidden");
    var zone = $("#uploadZone");
    if (zone) {
      zone.innerHTML = '<svg class="upload-icon" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg><p class="upload-title">点击或拖拽 CSV / Excel 文件至此处</p><p class="upload-hint">支持 .csv / .xls / .xlsx 格式，最大 50 MB</p>';
      zone.appendChild(fileInput);
    }
    // Quick-load: use /data-profile to get schema without re-uploading
    _origFetch(API_BASE + "/data-profile", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": "Bearer " + currentUser.token },
      body: JSON.stringify({ filename: filename }),
    }).then(function (r) { return r.json(); })
      .then(function (profile) {
        loadingIndicator.classList.add("hidden");
        // We need the full schema, not just profile. Use the stored schema from an upload response.
        // Since we only have the filename, let's just show the profile data
        // For proper resume, we'd need the original schema. Let's just set currentFilename.
        currentFilename = filename;
        // Try to get original schema from the data-profile and reconstruct
        profileData = profile;
        // Reconstruct a minimal schema from profile
        var cols = (profile.column_profiles || []).map(function (p) {
          return {
            name: p.name, dtype: p.dtype,
            missing_rate: p.missing_pct, missing_count: p.missing_count,
            unique_values: p.unique_count, stats: p.distribution || {},
          };
        });
        var schema = {
          filename: filename, total_rows: profile.rows, total_columns: profile.columns,
          columns: cols, sample_data: [],
        };
        currentSchema = schema;
        originalFilename = filename;
        originalSchema = JSON.parse(JSON.stringify(schema));
        renderSidebar(schema);
        updateTabLocks(schema);
        showUploadStatus("已恢复: " + filename, false);
        activeStep = "overview";
        profileData = profile;
        switchTab("understanding");
      }).catch(function (err) {
        loadingIndicator.classList.add("hidden");
        showToast("恢复失败: " + (err.message || ""), true);
      });
  }

  $("#btnSaveSettings").addEventListener("click", function () {
    var ak = globalApiKey.value.trim(), bu = globalBaseUrl.value.trim(), mn = globalModelName.value.trim();
    if (!ak || !bu || !mn) {
      settingsStatus.textContent = "请填写所有字段";
      settingsStatus.classList.remove("hidden", "text-green-600");
      settingsStatus.classList.add("text-red-500");
      return;
    }
    sessionStorage.setItem(STORAGE_KEYS.apiKey, ak);
    sessionStorage.setItem(STORAGE_KEYS.baseUrl, bu);
    sessionStorage.setItem(STORAGE_KEYS.modelName, mn);
    settingsStatus.textContent = "已保存";
    settingsStatus.classList.remove("hidden", "text-red-500");
    settingsStatus.classList.add("text-green-600");
    setTimeout(function () { settingsStatus.classList.add("hidden"); }, 2000);
  });

  $("#btnClearSettings").addEventListener("click", function () {
    sessionStorage.removeItem(STORAGE_KEYS.apiKey);
    sessionStorage.removeItem(STORAGE_KEYS.baseUrl);
    sessionStorage.removeItem(STORAGE_KEYS.modelName);
    globalApiKey.value = "";
    globalBaseUrl.value = "";
    globalModelName.value = "";
    settingsStatus.textContent = "配置已清空";
    settingsStatus.classList.remove("hidden", "text-red-500");
    settingsStatus.classList.add("text-green-600");
    setTimeout(function () { settingsStatus.classList.add("hidden"); }, 2000);
  });

  // =======================================================================
  // Tab Manager
  // =======================================================================
  function switchTab(tabName) {
    if (activeTab === tabName) return;
    var oldIdx = TAB_ORDER.indexOf(activeTab);
    var newIdx = TAB_ORDER.indexOf(tabName);
    if (oldIdx === -1 || newIdx === -1) return;

    var outDir = newIdx > oldIdx ? "left" : "right";
    var inDir  = newIdx > oldIdx ? "right" : "left";
    var oldPanel = $("#tab-" + activeTab);
    var newPanel = $("#tab-" + tabName);

    if (oldPanel && newPanel) {
      oldPanel.classList.add("tab-slide-out-" + outDir);
      oldPanel.addEventListener("animationend", function handler() {
        oldPanel.removeEventListener("animationend", handler);
        oldPanel.classList.add("hidden");
        oldPanel.classList.remove("tab-slide-out-" + outDir);
      }, { once: true });
    } else if (oldPanel) {
      oldPanel.classList.add("hidden");
    }

    newPanel.classList.remove("hidden");
    newPanel.classList.add("tab-slide-in-" + inDir);
    newPanel.addEventListener("animationend", function handler() {
      newPanel.removeEventListener("animationend", handler);
      newPanel.classList.remove("tab-slide-in-" + inDir);
    }, { once: true });

    $$(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
    var btn = document.querySelector('.tab-btn[data-tab="' + tabName + '"]');
    if (btn) btn.classList.add("active");

    activeTab = tabName;

    // Per-tab init
    if (tabName === "understanding" && currentFilename) {
      if (activeStep === "overview" && !profileData) loadProfile();
      if (activeStep === "cleaning") showCleaningStage();
    }
    if (tabName === "analysis" && currentFilename) setupAnalysisTab();
    if (tabName === "training" && currentSchema) populateTrainingSelectors();
    if (tabName === "sandbox") updateSandboxStatus();
  }

  $("#tabBar").addEventListener("click", function (e) {
    var btn = e.target.closest(".tab-btn");
    if (!btn || btn.classList.contains("disabled")) return;
    switchTab(btn.dataset.tab);
  });

  // =======================================================================
  // Conditional Tab Unlocking
  // =======================================================================
  function updateTabLocks(schema) {
    if (!schema || !schema.columns) return;
    var numCount = schema.columns.filter(function (c) { return c.dtype === "numeric"; }).length;
    var unlocked = numCount >= 2;
    var btnT = $("#tabBtnTraining"), btnS = $("#tabBtnSandbox");
    ["disabled", "data-tooltip"].forEach(function (a) { btnT.removeAttribute(a); btnS.removeAttribute(a); });
    btnT.classList.remove("disabled"); btnS.classList.remove("disabled");
    if (!unlocked) {
      btnT.classList.add("disabled");
      btnT.setAttribute("data-tooltip", "需要至少 2 列数值型数据（当前 " + numCount + " 列）");
      btnS.classList.add("disabled");
      btnS.setAttribute("data-tooltip", "需要至少 2 列数值型数据（当前 " + numCount + " 列）");
    }
  }

  // =======================================================================
  // TAB 1: File Upload (preserved)
  // =======================================================================
  var uploadZone = $("#uploadZone"), fileInput = $("#fileInput");
  var uploadStatus = $("#uploadStatus"), loadingIndicator = $("#loadingIndicator");
  var sidebarOverview = $("#sidebarOverview"), sidebarColumns = $("#sidebarColumns");

  function validateFile(file) {
    if (!file) return { ok: false, error: "未检测到文件。" };
    var ext = "." + file.name.split(".").pop().toLowerCase();
    if (ALLOWED_EXTENSIONS.indexOf(ext) === -1) return { ok: false, error: "不支持的文件格式。" };
    if (file.size > MAX_FILE_SIZE) return { ok: false, error: "文件过大（上限 50 MB）。" };
    return { ok: true };
  }

  function uploadFile(file) {
    loadingIndicator.classList.remove("hidden");
    uploadZone.classList.add("uploading");
    var fd = new FormData(); fd.append("file", file);
    fetch(API_BASE + "/upload", { method: "POST", body: fd })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (b) { throw new Error(b.detail || "服务器错误"); });
        return r.json();
      })
      .then(function (schema) {
        loadingIndicator.classList.add("hidden");
        uploadZone.classList.remove("uploading");
        currentSchema = schema;
        currentFilename = schema.filename;
        originalFilename = schema.filename;
        originalSchema = JSON.parse(JSON.stringify(schema));  // deep copy
        profileData = null;
        analysisHistory = [];
        activeAnalysisId = null;

        renderSidebar(schema);
        updateTabLocks(schema);
        showUploadStatus("解析完成 — " + schema.total_rows.toLocaleString() + " 行 x " + schema.total_columns + " 列", false);

        // Auto-switch to understanding tab
        activeStep = "overview";
        switchTab("understanding");
      })
      .catch(function (err) {
        loadingIndicator.classList.add("hidden");
        uploadZone.classList.remove("uploading");
        showErrorModal("上传失败", err.message || "请检查网络连接和后端是否正常启动");
        showUploadStatus("上传失败: " + err.message, true);
      });
  }

  function showUploadStatus(text, isErr) {
    uploadStatus.textContent = text;
    uploadStatus.className = "mt-4 text-sm " + (isErr ? "text-red-500" : "text-green-600");
    uploadStatus.classList.remove("hidden");
  }

  function renderSidebar(schema) {
    sidebarOverview.innerHTML =
      '<div class="grid grid-cols-2 gap-2 mb-3">' +
        '<div class="bg-gray-50 rounded-lg p-2.5 text-center"><p class="text-base font-bold text-gray-900">' + schema.total_rows.toLocaleString() + '</p><p class="text-[10px] text-gray-400 uppercase">行数</p></div>' +
        '<div class="bg-gray-50 rounded-lg p-2.5 text-center"><p class="text-base font-bold text-gray-900">' + schema.total_columns + '</p><p class="text-[10px] text-gray-400 uppercase">列数</p></div></div>';
    var h = '<h3 class="text-xs font-semibold text-gray-500 uppercase mt-3 mb-1.5">字段</h3><ul class="space-y-0.5">';
    schema.columns.forEach(function (c) {
      var tag = dtypeTag(c.dtype);
      var miss = "";
      if (c.missing_rate > 0) {
        var p = (c.missing_rate * 100).toFixed(1);
        miss = '<span class="missing-bar-track"><span class="missing-bar-fill" style="width:' + p + '%"></span></span>';
      }
      h += '<li class="flex items-center justify-between py-1 px-1.5 rounded text-xs hover:bg-gray-50">' +
             '<span class="text-gray-700 truncate mr-1">' + escapeHtml(c.name) + '</span>' +
             '<span class="flex items-center gap-1 flex-shrink-0">' + tag + miss + '</span></li>';
    });
    h += '</ul>';
    sidebarColumns.innerHTML = h;
  }

  // Upload events
  uploadZone.addEventListener("click", function (e) {
    if (!currentUser || !currentUser.token) { showToast("请先在左侧登录"); return; }
    if (e.target !== fileInput) fileInput.click();
  });
  fileInput.addEventListener("change", function () {
    var f = fileInput.files[0]; if (!f) return;
    var v = validateFile(f); if (!v.ok) { showErrorModal("文件无效", v.error); return; }
    uploadFile(f);
  });
  uploadZone.addEventListener("dragover", function (e) { e.preventDefault(); uploadZone.classList.add("drag-over"); });
  uploadZone.addEventListener("dragleave", function () { uploadZone.classList.remove("drag-over"); });
  uploadZone.addEventListener("drop", function (e) {
    e.preventDefault(); uploadZone.classList.remove("drag-over");
    if (!currentUser || !currentUser.token) { showToast("请先在左侧登录"); return; }
    var f = e.dataTransfer.files[0]; if (!f) return;
    var v = validateFile(f); if (!v.ok) { showErrorModal("文件无效", v.error); return; }
    uploadFile(f);
  });
  ["dragenter","dragover","dragleave","drop"].forEach(function (ev) {
    document.body.addEventListener(ev, function (e) { e.preventDefault(); e.stopPropagation(); });
  });

  // =======================================================================
  // TAB 2: Data Understanding — Stepper
  // =======================================================================
  var stepperBar = $("#stepperBar");
  var stageOverview = $("#stage-overview"), stageCleaning = $("#stage-cleaning");
  var overviewEmpty = $("#overviewEmpty"), overviewContent = $("#overviewContent");

  stepperBar.addEventListener("click", function (e) {
    var step = e.target.closest(".stepper-step");
    if (!step) return;
    var stepName = step.dataset.step;
    if (!currentFilename) return;
    setActiveStep(stepName);
  });

  function setActiveStep(stepName) {
    activeStep = stepName;
    $$(".stepper-step").forEach(function (s) { s.classList.remove("active"); });
    var btn = document.querySelector('.stepper-step[data-step="' + stepName + '"]');
    if (btn) btn.classList.add("active");

    if (stepName === "overview") {
      stageOverview.classList.remove("hidden");
      stageCleaning.classList.add("hidden");
      if (!profileData) loadProfile();
    } else {
      stageOverview.classList.add("hidden");
      stageCleaning.classList.remove("hidden");
      showCleaningStage();
    }
  }

  // ---- Stage 2a: Data Overview ----
  function loadProfile() {
    if (!currentFilename) return;
    overviewEmpty.classList.add("hidden");
    overviewContent.classList.remove("hidden");
    showOverviewLoading();

    fetch(API_BASE + "/data-profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: currentFilename }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        profileData = data;
        renderOverview(data);
      })
      .catch(function (err) {
        showToast("数据画像加载失败：" + err.message);
        overviewContent.classList.add("hidden");
        overviewEmpty.classList.remove("hidden");
      });
  }

  function showOverviewLoading() {
    $("#overviewCards").innerHTML = '<div class="col-span-full text-center py-4" style="color:#94A3B8;"><div class="spinner w-5 h-5 mb-2"></div><p class="text-xs">正在生成数据画像...</p></div>';
    $("#columnCards").innerHTML = "";
    $("#sampleTable").innerHTML = "";
  }

  function renderOverview(data) {
    // Summary cards
    var cards = [
      { label: "行数", value: data.rows.toLocaleString() },
      { label: "列数", value: data.columns },
      { label: "完整度", value: (data.completeness * 100).toFixed(1) + "%" },
      { label: "重复行", value: data.duplicate_rows },
    ];
    $("#overviewCards").innerHTML = cards.map(function (c) {
      return '<div class="overview-card"><div class="overview-card-value">' + c.value + '</div><div class="overview-card-label">' + c.label + '</div></div>';
    }).join("");

    // Field type pie
    renderFieldTypeChart(data.column_profiles);

    // AI summary (skip if empty — will be populated via LLM)
    if (isLlmConfigured()) generateAiSummary();
    else $("#aiSummaryContent").innerHTML = '<p class="text-xs" style="color:#94A3B8;">请在左侧「LLM 配置」中填写 API 信息后自动生成数据摘要。</p>';

    // Sample table
    renderSampleTable(currentSchema.sample_data, "head");

    // Column detail cards
    renderColumnCards(data.column_profiles);

  }

  function generateAiSummary() {
    var config = getLlmConfig();
    var schema = currentSchema;
    // Use legacy ask_data approach for simplicity
    fetch(API_BASE + "/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: "请用一段话（100字以内）总结这份数据：它包含什么信息，适合回答什么问题，不适合回答什么问题。",
        table_schema: schema,
        api_key: config.api_key, base_url: config.base_url, model_name: config.model_name,
      }),
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.answer) $("#aiSummaryContent").innerHTML = '<p class="text-sm leading-relaxed">' + escapeHtml(d.answer) + '</p>';
        else $("#aiSummaryContent").innerHTML = '<p class="text-xs" style="color:#94A3B8;">摘要生成失败：' + escapeHtml(d.error || "") + '</p>';
      })
      .catch(function () {
        $("#aiSummaryContent").innerHTML = '<p class="text-xs" style="color:#94A3B8;">摘要生成失败</p>';
      });
  }

  function renderFieldTypeChart(profiles) {
    var counts = { numeric: 0, categorical: 0, datetime: 0, text: 0, boolean: 0 };
    profiles.forEach(function (p) { if (counts[p.dtype] !== undefined) counts[p.dtype]++; });
    var dom = $("#fieldTypeChart"); dom.innerHTML = "";
    if (typeof echarts === "undefined") return;
    var chart = echarts.init(dom);
    chart.setOption({
      tooltip: { trigger: "item" },
      series: [{
        type: "pie", radius: ["55%","75%"], center: ["50%","50%"],
        data: [
          { name: "数值", value: counts.numeric, itemStyle: { color: "#3B82F6" } },
          { name: "分类", value: counts.categorical, itemStyle: { color: "#8B5CF6" } },
          { name: "日期", value: counts.datetime, itemStyle: { color: "#06B6D4" } },
          { name: "文本", value: counts.text, itemStyle: { color: "#F59E0B" } },
          { name: "布尔", value: counts.boolean, itemStyle: { color: "#10B981" } },
        ].filter(function (d) { return d.value > 0; }),
        label: { fontSize: 10 },
      }],
    });
  }

  function renderSampleTable(sampleData, mode) {
    if (!sampleData || !sampleData.length) { $("#sampleTable").innerHTML = '<p class="text-xs" style="color:#94A3B8;">无样本数据。</p>'; return; }
    var cols = Object.keys(sampleData[0]);
    var h = '<table style="width:100%; font-size:0.75rem; border-collapse:collapse;"><thead><tr>';
    cols.forEach(function (c) { h += '<th style="text-align:left;padding:0.375rem 0.5rem;font-size:0.6875rem;color:#94A3B8;border-bottom:1px solid #E2E8F0;white-space:nowrap;">' + escapeHtml(c) + '</th>'; });
    h += '</tr></thead><tbody>';
    sampleData.forEach(function (row) {
      h += '<tr>';
      cols.forEach(function (c) { h += '<td style="padding:0.375rem 0.5rem;border-bottom:1px solid #F1F5F9;white-space:nowrap;color:#475569;">' + escapeHtml(String(row[c] != null ? row[c] : "")) + '</td>'; });
      h += '</tr>';
    });
    h += '</tbody></table>';
    $("#sampleTable").innerHTML = h;
  }

  // Sample tab switcher
  $$("#sampleTabBtns button, .sample-tab-btn")[0] && $$("#sampleTabBtns button, .sample-tab-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      $$(".sample-tab-btn").forEach(function (b) { b.classList.remove("active"); });
      this.classList.add("active");
      // Always show head for simplicity; we rely on schema.sample_data
      var schema = currentSchema;
      if (schema && schema.sample_data) renderSampleTable(schema.sample_data, this.dataset.sample);
    });
  });

  // Delegate sample-tab-btn clicks from the overview area
  $("#stage-overview").addEventListener("click", function (e) {
    var btn = e.target.closest(".sample-tab-btn");
    if (!btn) return;
    $$(".sample-tab-btn").forEach(function (b) { b.classList.remove("active"); });
    btn.classList.add("active");
    if (currentSchema && currentSchema.sample_data) renderSampleTable(currentSchema.sample_data, btn.dataset.sample);
  });

  function renderColumnCards(profiles) {
    var html = "";
    profiles.forEach(function (p) {
      var sparklineHtml = "";
      if (p.sparkline && p.sparkline.length > 0) {
        var maxVal = Math.max.apply(null, p.sparkline);
        sparklineHtml = '<div class="flex items-end gap-px mt-2" style="height:32px;">' +
          p.sparkline.map(function (v) {
            var h = maxVal > 0 ? Math.max(2, (v / maxVal) * 32) : 2;
            return '<span class="sparkline-bar" style="height:' + h + 'px;"></span>';
          }).join("") + '</div>';
      }
      var distStr = "";
      if (p.distribution) {
        distStr = '<div class="text-[10px] mt-1" style="color:#94A3B8;">范围: ' + (p.distribution.min != null ? p.distribution.min.toFixed(1) : "-") + ' ~ ' + (p.distribution.max != null ? p.distribution.max.toFixed(1) : "-") + ' | 中位数: ' + p.distribution.median.toFixed(2) + '</div>';
      }
      var topStr = "";
      if (p.top_values && p.top_values.length > 0) {
        topStr = '<div class="text-[10px] mt-1" style="color:#94A3B8;">' + p.top_values.slice(0, 3).map(function (t) { return escapeHtml(t.name) + ":" + t.count; }).join(" | ") + '</div>';
      }
      var outlierTag = "";
      if (p.outlier_count > 0) outlierTag = '<span class="inline-block text-[10px] px-1 py-0.5 rounded-full ml-1" style="background:#FEF3C7;color:#D97706;">' + p.outlier_count + ' 异常</span>';
      html += '<div class="column-detail-card">' +
        '<div class="column-detail-name">' + escapeHtml(p.name) + outlierTag + '</div>' +
        '<div class="flex items-center gap-2 mb-1">' + dtypeTag(p.dtype) +
          '<span class="text-[10px]" style="color:#94A3B8;">缺失 ' + (p.missing_pct * 100).toFixed(1) + '% | 唯一值 ' + p.unique_count + '</span>' +
        '</div>' + distStr + topStr + sparklineHtml + '</div>';
    });
    $("#columnCards").innerHTML = html;
  }

  // ---- Stage 2b: Data Cleaning ----
  var cleaningPanels = $("#cleaningPanels"), cleaningLogEntries = $("#cleaningLogEntries");
  var cleaningLog = $("#cleaningLog"), cleaningResult = $("#cleaningResult");

  function showCleaningStage() {
    var cleaningEmpty = $("#cleaningEmpty"), cleaningContent = $("#cleaningContent");
    if (!currentFilename) {
      cleaningEmpty.classList.remove("hidden"); cleaningContent.classList.add("hidden"); return;
    }
    cleaningEmpty.classList.add("hidden"); cleaningContent.classList.remove("hidden");
    detectIssues();
  }

  function detectIssues() {
    if (!profileData) {
      fetch(API_BASE + "/data-profile", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: currentFilename }),
      })
        .then(function (r) { return r.json(); })
        .then(function (data) { profileData = data; renderCleaningPanels(data); })
        .catch(function () { cleaningPanels.innerHTML = '<p class="text-sm text-red-500">数据检测失败</p>'; });
    } else {
      renderCleaningPanels(profileData);
    }
  }

  $("#btnRefreshCleaning").addEventListener("click", function () {
    profileData = null; detectIssues();
  });

  function renderCleaningPanels(data) {
    var profiles = data.column_profiles;
    var issues = data.quality_issues || {};

    // 1. Missing values
    var highMissing = issues.high_missing || [];
    var hasMissing = profiles.filter(function (p) { return p.missing_pct > 0; });

    // 2. Outliers
    var hasOutliers = profiles.filter(function (p) { return p.outlier_count > 0; });

    // 3. Constant columns
    var constantCols = issues.constant_columns || [];

    // 4. Type suggestions
    var typeSugs = issues.type_suggestions || [];

    var panelsHtml = "";

    // --- Missing panel ---
    panelsHtml += buildCleaningPanel("缺失值处理", hasMissing.length,
      '<div class="text-xs mb-2" style="color:#64748B;">共 ' + hasMissing.length + ' 列存在缺失值。AI 推荐：数值列用中位数填充，分类列用众数填充。</div>' +
      '<div class="overflow-x-auto"><table class="analysis-stats-table"><thead><tr><th>列名</th><th>缺失率</th><th>缺失数</th><th>处理方式</th></tr></thead><tbody>' +
      hasMissing.map(function (p) {
        var recMethod = (p.dtype === "numeric") ? "median" : "mode";
        return '<tr><td>' + escapeHtml(p.name) + '</td><td>' + (p.missing_pct * 100).toFixed(1) + '%</td><td>' + p.missing_count + '</td>' +
          '<td><select class="clean-select text-xs" data-clean="fill_missing" data-col="' + escapeHtml(p.name) + '">' +
            '<option value="median" ' + (recMethod === "median" ? "selected" : "") + '>中位数填充</option>' +
            '<option value="mean">均值填充</option>' +
            '<option value="mode">众数填充</option>' +
            '<option value="drop_rows">删除含缺失行</option>' +
            '<option value="keep">保留</option>' +
          '</select></td></tr>';
      }).join("") +
      '</tbody></table></div>',
      hasMissing.length > 0 ? "warning" : "ok");

    // --- Outlier panel ---
    panelsHtml += buildCleaningPanel("异常值处理", hasOutliers.length,
      '<div class="text-xs mb-2" style="color:#64748B;">共 ' + hasOutliers.length + ' 列检测到异常值（IQR 方法）。AI 推荐：保留异常值，仅标记不处理。</div>' +
      '<div class="overflow-x-auto"><table class="analysis-stats-table"><thead><tr><th>列名</th><th>异常值数</th><th>占比</th><th>处理方式</th></tr></thead><tbody>' +
      hasOutliers.map(function (p) {
        var opct = p.outlier_count > 0 && p.distribution ? (p.outlier_count / (p.missing_count > 0 ? 1 : 1) * 100).toFixed(1) : "—"; // rough
        return '<tr><td>' + escapeHtml(p.name) + '</td><td>' + p.outlier_count + '</td><td>' + opct + '%</td>' +
          '<td><select class="clean-select text-xs" data-clean="handle_outliers" data-col="' + escapeHtml(p.name) + '">' +
            '<option value="keep" selected>保留</option>' +
            '<option value="cap">截断到边界</option>' +
            '<option value="remove">剔除</option>' +
          '</select></td></tr>';
      }).join("") +
      '</tbody></table></div>',
      hasOutliers.length > 0 ? "warning" : "ok");

    // --- Duplicates panel ---
    panelsHtml += buildCleaningPanel("重复行", data.duplicate_rows,
      '<div class="text-xs mb-2" style="color:#64748B;">检测到 ' + data.duplicate_rows + ' 行重复数据。AI 推荐：删除重复行。</div>' +
      (data.duplicate_rows > 0 ? '<select class="clean-select text-xs" data-clean="drop_duplicates"><option value="first" selected>删除重复（保留首次）</option><option value="keep">保留不处理</option></select>' : ''),
      data.duplicate_rows > 0 ? "warning" : "ok");

    // --- Constant columns panel ---
    panelsHtml += buildCleaningPanel("常量列", constantCols.length,
      '<div class="text-xs mb-2" style="color:#64748B;">以下列的所有值相同：' + constantCols.map(escapeHtml).join(", ") + '。AI 推荐：删除。</div>' +
      (constantCols.length > 0 ? '<select class="clean-select text-xs" data-clean="drop_constant_columns"><option value="drop" selected>删除常量列</option><option value="keep">保留</option></select>' : ''),
      constantCols.length > 0 ? "warning" : "ok");

    // --- Type suggestions panel ---
    panelsHtml += buildCleaningPanel("数据类型建议", typeSugs.length,
      '<div class="text-xs mb-2" style="color:#64748B;">' + (typeSugs.length > 0 ? '以下列可能存储为错误类型。AI 推荐：自动转换。' : '未发现类型问题。') + '</div>' +
      (typeSugs.length > 0 ? '<div class="overflow-x-auto"><table class="analysis-stats-table"><thead><tr><th>列名</th><th>当前类型</th><th>建议类型</th></tr></thead><tbody>' +
        typeSugs.map(function (t) {
          return '<tr><td>' + escapeHtml(t.column) + '</td><td>' + t.current + '</td><td>' + t.suggested + '</td></tr>';
        }).join("") + '</tbody></table></div>' : ''),
      typeSugs.length > 0 ? "warning" : "ok");

    cleaningPanels.innerHTML = panelsHtml;
    bindCleaningPanelToggles();
  }

  function buildCleaningPanel(title, issueCount, bodyHtml, badgeType) {
    var badgeCls = badgeType === "warning" ? "warning" : "ok";
    return '<div class="cleaning-panel">' +
      '<div class="cleaning-panel-header">' +
        '<div class="cleaning-panel-header-left">' +
          '<span class="cleaning-panel-title">' + title + '</span>' +
          '<span class="cleaning-panel-badge ' + badgeCls + '">' + (issueCount || "无") + '</span>' +
        '</div>' +
        '<svg class="cleaning-panel-chevron" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" d="M6 9l6 6 6-6"/></svg>' +
      '</div>' +
      '<div class="cleaning-panel-body hidden">' + bodyHtml + '</div>' +
    '</div>';
  }

  function bindCleaningPanelToggles() {
    $$(".cleaning-panel-header").forEach(function (hdr) {
      hdr.addEventListener("click", function () {
        this.classList.toggle("expanded");
        var body = this.nextElementSibling;
        if (body) body.classList.toggle("hidden");
      });
    });
  }

  // Apply cleaning
  $("#btnApplyCleaning").addEventListener("click", function () {
    var ops = [];
    $$(".clean-select").forEach(function (sel) {
      var cleanType = sel.dataset.clean;
      var col = sel.dataset.col;
      var val = sel.value;
      if (val === "keep") return;

      if (cleanType === "fill_missing") {
        if (val === "drop_rows") {
          // Handled differently — we'd need to remove rows; for now skip or add a drop operation
          ops.push({ type: "fill_missing", column: col, method: "median" });
        } else {
          ops.push({ type: "fill_missing", column: col, method: val });
        }
      } else if (cleanType === "handle_outliers") {
        ops.push({ type: "handle_outliers", column: col, method: val });
      } else if (cleanType === "drop_duplicates") {
        if (val === "first") ops.push({ type: "drop_duplicates", keep: "first" });
      } else if (cleanType === "drop_constant_columns") {
        if (val === "drop") ops.push({ type: "drop_constant_columns", columns: [] });
      }
    });

    if (ops.length === 0) { showToast("未选择任何清洗操作"); return; }

    var btn = $("#btnApplyCleaning");
    btn.disabled = true; btn.textContent = "正在清洗...";

    fetch(API_BASE + "/clean-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: currentFilename, operations: ops }),
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        btn.disabled = false; btn.textContent = "应用清洗";
        // Update filename & schema to cleaned version
        currentFilename = d.new_filename;
        currentSchema = d.schema;
        profileData = null;

        // Show log
        cleaningLog.classList.remove("hidden");
        var summary = d.summary;
        cleaningLogEntries.innerHTML = summary.log.map(function (l) { return '<div>' + escapeHtml(l) + '</div>'; }).join("");

        cleaningResult.textContent = "清洗完成: " + summary.rows_before.toLocaleString() + " 行 -> " + summary.rows_after.toLocaleString() + " 行 | " + summary.cols_before + " 列 -> " + summary.cols_after + " 列";
        cleaningResult.classList.remove("hidden", "text-red-500");
        cleaningResult.classList.add("text-green-600");

        // Refresh overview after cleaning
        renderSidebar(d.schema);
        updateTabLocks(d.schema);

        showToast("数据清洗完成");
      })
      .catch(function (err) {
        btn.disabled = false; btn.textContent = "应用清洗";
        showToast("清洗失败：" + err.message);
      });
  });

  // =======================================================================
  // TAB 3: AI Analysis — Notebook Mode
  // =======================================================================
  var analysisInput = $("#analysisInput"), btnSendAnalysis = $("#btnSendAnalysis");
  var analysisCardsArea = $("#analysisCardsArea"), notebookTimeline = $("#notebookTimeline");
  var analysisEmptyState = $("#analysisEmptyState");

  function setupAnalysisTab() {
    if (analysisHistory.length === 0) {
      analysisEmptyState.classList.remove("hidden");
    } else {
      analysisEmptyState.classList.add("hidden");
    }
    updateTimeline();
  }

  function updateTimeline() {
    if (analysisHistory.length === 0) {
      notebookTimeline.innerHTML = '<p class="text-xs" style="color:#94A3B8;">暂无分析记录，输入问题开始探索数据。</p>';
      return;
    }
    var html = "";
    analysisHistory.forEach(function (h, i) {
      var activeClass = (h.id === activeAnalysisId) ? " active" : "";
      html += '<div class="notebook-timeline-item' + activeClass + '" data-analysis-id="' + h.id + '">' +
        '<div class="notebook-timeline-question">' + escapeHtml(h.question) + '</div>' +
        '<div class="notebook-timeline-time">' + h.time + '</div></div>';
    });
    notebookTimeline.innerHTML = html;
  }

  notebookTimeline.addEventListener("click", function (e) {
    var item = e.target.closest(".notebook-timeline-item");
    if (!item) return;
    var id = item.dataset.analysisId;
    switchToAnalysisCard(id);
  });

  $("#btnNewAnalysis").addEventListener("click", function () {
    activeAnalysisId = null;
    analysisEmptyState.classList.remove("hidden");
    // Hide all cards
    $$(".analysis-card").forEach(function (c) { c.classList.add("hidden"); });
    updateTimeline();
  });

  function switchToAnalysisCard(id) {
    activeAnalysisId = id;
    $$(".analysis-card").forEach(function (c) { c.classList.add("hidden"); });
    var card = $("#analysis-card-" + id);
    if (card) card.classList.remove("hidden");
    $$(".notebook-timeline-item").forEach(function (i) { i.classList.remove("active"); });
    var item = document.querySelector('.notebook-timeline-item[data-analysis-id="' + id + '"]');
    if (item) item.classList.add("active");
    analysisEmptyState.classList.add("hidden");
  }

  // Send analysis question
  function sendAnalysis() {
    if (analysisLoading) return;
    var q = analysisInput.value.trim(); if (!q) return;
    if (!isLlmConfigured()) { showToast("请先在「数据接入」左侧配置 LLM"); return; }
    if (!currentFilename || !currentSchema) { showToast("请先在「数据接入」上传文件"); return; }

    analysisLoading = true; btnSendAnalysis.disabled = true;
    analysisEmptyState.classList.add("hidden");

    // Create analysis session
    var analysisId = "a" + Date.now();
    var now = new Date().toLocaleTimeString();
    activeAnalysisId = analysisId;

    // Build the analysis card
    var card = document.createElement("div");
    card.className = "analysis-card";
    card.id = "analysis-card-" + analysisId;
    card.innerHTML =
      '<div class="analysis-card-header">' +
        '<div class="analysis-card-question">' + escapeHtml(q) + '</div>' +
        '<div class="analysis-card-translation hidden" id="card-translation-' + analysisId + '"></div>' +
        '<div class="analysis-card-plan text-xs mt-1 hidden" style="color:#64748B;" id="card-plan-' + analysisId + '"></div>' +
      '</div>' +
      '<div class="analysis-card-body" id="card-body-' + analysisId + '">' +
        '<div class="sse-status-bar"><span class="sse-status-dot"></span><span id="card-status-' + analysisId + '">正在理解问题...</span></div>' +
      '</div>' +
      '<div class="analysis-card-interpretation hidden" id="card-interp-' + analysisId + '"></div>';
    analysisCardsArea.appendChild(card);
    card.scrollIntoView({ behavior: "smooth" });

    // Hide other cards
    $$(".analysis-card").forEach(function (c) { c.classList.add("hidden"); });
    card.classList.remove("hidden");

    // Push to history
    analysisHistory.push({ id: analysisId, question: q, time: now });
    updateTimeline();

    analysisInput.value = "";

    // Start SSE
    var config = getLlmConfig();
    var payload = {
      question: q,
      filename: currentFilename,
      table_schema: currentSchema,
      llm_config: { api_key: config.api_key, base_url: config.base_url, model_name: config.model_name },
    };

    fetch(API_BASE + "/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (response) {
      if (!response.ok) {
        return response.json().then(function (err) { throw new Error(err.detail || "请求失败"); });
      }
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      var chunkCount = 0;
      function process() {
        reader.read().then(function (result) {
          if (result.done) {
            console.log("[SSE] Stream ended after", chunkCount, "chunks");
            analysisLoading = false; btnSendAnalysis.disabled = false;
            // Process any remaining buffer
            if (buffer.trim()) {
              var remaining = buffer.split("\n");
              var evt = "";
              remaining.forEach(function (line) {
                line = line.replace(/\r$/, "");
                if (line.startsWith("event: ")) { evt = line.slice(7).trim(); }
                else if (line.startsWith("data: ") && evt) {
                  try { handleSseEvent(analysisId, evt, JSON.parse(line.slice(6))); } catch (e) {}
                }
              });
            }
            return;
          }
          chunkCount++;
          buffer += decoder.decode(result.value, { stream: true });

          // Parse complete SSE events, keeping incomplete line in buffer
          var lines = buffer.split("\n");
          // The last element may be incomplete — keep it in buffer
          buffer = lines.pop() || "";
          var currentEvent = "";
          for (var i = 0; i < lines.length; i++) {
            var line = lines[i].replace(/\r$/, "");
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              if (!currentEvent) continue;
              var dataStr = line.slice(6);
              try {
                var data = JSON.parse(dataStr);
                handleSseEvent(analysisId, currentEvent, data);
              } catch (e) { console.warn("SSE parse error:", e, dataStr.slice(0, 80)); }
            }
            // Empty lines (event separator) — keep currentEvent for next event
          }
          process();
        }).catch(function (err) {
          analysisLoading = false; btnSendAnalysis.disabled = false;
          console.error("SSE stream error:", err);
        });
      }
      process();
    }).catch(function (err) {
      analysisLoading = false; btnSendAnalysis.disabled = false;
      showToast("分析请求失败：" + err.message);
    });
  }

  function handleSseEvent(analysisId, eventType, data) {
    console.log("[SSE]", eventType, data);
    switch (eventType) {
      case "translation":
        var transEl = $("#card-translation-" + analysisId);
        if (transEl) { transEl.classList.remove("hidden"); transEl.textContent = data.text || ""; }
        break;

      case "plan":
        var planEl = $("#card-plan-" + analysisId);
        var hasActions = data.actions && data.actions.length > 0;
        if (planEl) {
          planEl.classList.remove("hidden");
          if (!hasActions && data.plan_text) {
            // Chat mode — show answer directly
            planEl.textContent = data.plan_text;
            planEl.style.color = "#334155";
          } else {
            planEl.textContent = data.plan_text || "";
          }
        }
        console.log("[SSE] Plan received, actions:", data.actions ? data.actions.length : 0);
        if (hasActions) {
          var body = $("#card-body-" + analysisId);
          if (body) {
            var statusBar = body.querySelector(".sse-status-bar");
            if (statusBar) {
              statusBar.innerHTML = '<span class="sse-status-dot"></span><span id="card-status-' + analysisId + '">正在执行分析...</span><div class="sse-progress-track" id="card-progress-' + analysisId + '"><div class="sse-progress-fill" style="width:0%"></div></div>';
            }
          }
        }
        break;

      case "progress":
        var total = data.total || 1;
        var step = data.step || 1;
        var pct = Math.round(step / total * 100);
        var fill = $("#card-progress-" + analysisId);
        if (fill) fill.style.width = pct + "%";
        var statusEl = $("#card-status-" + analysisId);
        if (statusEl) statusEl.textContent = (data.message || "") + " (" + step + "/" + total + ")";
        break;

      case "result":
        console.log("[SSE] Rendering result:", data.function, data.chart_type);
        renderAnalysisResult(analysisId, data);
        break;

      case "interpretation":
        var interpEl = $("#card-interp-" + analysisId);
        if (interpEl) {
          interpEl.classList.remove("hidden");
          interpEl.textContent += data.text || "";
          interpEl.classList.add("streaming-cursor");
        }
        break;

      case "done":
        console.log("[SSE] Done");
        analysisLoading = false; btnSendAnalysis.disabled = false;
        var interpDone = $("#card-interp-" + analysisId);
        if (interpDone) interpDone.classList.remove("streaming-cursor");
        var bodyEl = $("#card-body-" + analysisId);
        if (bodyEl) {
          var sb = bodyEl.querySelector(".sse-status-bar");
          if (sb) sb.style.display = "none";
        }
        break;

      case "error":
        console.error("[SSE] Error:", data);
        showErrorModal("分析异常", data.message || "分析过程出现未知错误，请检查 LLM 配置或重试。");
        var errStatus = $("#card-status-" + analysisId);
        if (errStatus) errStatus.textContent = "出错: " + (data.message || "未知错误");
        var errProgress = $("#card-progress-" + analysisId);
        if (errProgress) errProgress.parentElement.style.background = "#FEE2E2";
        analysisLoading = false; btnSendAnalysis.disabled = false;
        break;

      case "status":
        break;  // status text handled inline via card-status span
    }
  }

  function renderAnalysisResult(analysisId, data) {
    var body = $("#card-body-" + analysisId);
    if (!body) { console.warn("[SSE] card-body not found:", analysisId); return; }

    console.log("[SSE] renderAnalysisResult:", data.function, "chart:", data.chart_type, "has chart_data:", !!data.chart_data, "error:", data.error || "none");

    // Show error if present
    if (data.error) {
      var errDiv = document.createElement("div");
      errDiv.className = "text-xs mb-2 p-2 rounded";
      errDiv.style.cssText = "background:#FEF2F2; border:1px solid #FECACA; color:#991B1B;";
      errDiv.textContent = "[" + data.function + "] " + data.error;
      body.appendChild(errDiv);
    }

    var chartWrap = document.createElement("div");
    chartWrap.className = "analysis-chart-wrap";
    chartWrap.id = "chart-" + analysisId + "-" + data.step;
    chartWrap.style.outline = "1px dashed #E2E8F0";
    chartWrap.style.minHeight = "300px";
    chartWrap.style.width = "100%";
    body.appendChild(chartWrap);
    console.log("[SSE] Chart wrap created:", chartWrap.id, "parent:", body.id, "visible:", chartWrap.offsetWidth + "x" + chartWrap.offsetHeight);

    // Stats table
    if (data.stats && Object.keys(data.stats).length > 0) {
      var tableWrap = document.createElement("div");
      var rows = Object.entries(data.stats).filter(function (e) { return e[0] !== "error" && typeof e[1] !== "object"; });
      if (rows.length > 0) {
        tableWrap.innerHTML = '<table class="analysis-stats-table"><tbody>' +
          rows.map(function (r) {
            return '<tr><td class="font-medium" style="color:#64748B;">' + escapeHtml(String(r[0])) + '</td><td>' + escapeHtml(String(r[1])) + '</td></tr>';
          }).join("") + '</tbody></table>';
        body.appendChild(tableWrap);
      }
    }

    // Table data
    if (data.table && data.table.length > 0) {
      var tableDiv = document.createElement("div");
      tableDiv.className = "overflow-x-auto mb-3";
      var th = '<table class="analysis-stats-table"><tbody>';
      data.table.forEach(function (row) {
        th += '<tr>' + row.map(function (cell) { return '<td>' + escapeHtml(String(cell)) + '</td>'; }).join("") + '</tr>';
      });
      th += '</tbody></table>';
      tableDiv.innerHTML = th;
      body.appendChild(tableDiv);
    }

    // Render chart via ECharts
    setTimeout(function () {
      renderAnalysisChart(chartWrap.id, data);
    }, 50);
  }

  function renderAnalysisChart(domId, data) {
    var dom = document.getElementById(domId);
    if (!dom) { console.warn("[SSE] Chart DOM not found:", domId); return; }
    if (typeof echarts === "undefined") { console.warn("[SSE] ECharts not loaded"); return; }
    if (dom.offsetWidth === 0 || dom.offsetHeight === 0) {
      console.warn("[SSE] Chart DOM has zero dimensions:", domId, dom.offsetWidth, dom.offsetHeight);
    }
    try {
      var chart = echarts.init(dom);
    } catch (e) {
      console.error("[SSE] ECharts init failed:", e);
      return;
    }
    var chartType = data.chart_type || "table";
    var chartData = data.chart_data || {};

    var option = null;

    switch (chartType) {
      case "histogram":
      case "bar":
        option = {
          tooltip: { trigger: "axis" },
          grid: { left: "8%", right: "4%", top: "8%", bottom: "8%" },
          xAxis: { type: "category", data: (chartData.bins || chartData.labels || []).map(function (v) { return typeof v === "number" ? v.toFixed(1) : v; }), axisLabel: { fontSize: 10, rotate: 30 } },
          yAxis: { type: "value", axisLabel: { fontSize: 10 } },
          series: [{ type: "bar", data: chartData.counts || chartData.values || [], itemStyle: { color: "#0052CC", borderRadius: [2,2,0,0] } }],
        };
        break;

      case "grouped_bar":
        option = {
          tooltip: { trigger: "axis" },
          legend: { data: ["分段", "整体"], textStyle: { fontSize: 10 }, top: 0 },
          grid: { left: "3%", right: "4%", top: "12%", bottom: "8%" },
          xAxis: { type: "category", data: chartData.labels || [], axisLabel: { fontSize: 10, rotate: 30 } },
          yAxis: { type: "value", axisLabel: { fontSize: 10 } },
          series: [
            { name: "分段", type: "bar", data: chartData.segment_values || [], itemStyle: { color: "#0052CC" } },
            { name: "整体", type: "bar", data: chartData.overall_values || [], itemStyle: { color: "#94A3B8" } },
          ],
        };
        break;

      case "horizontal_bar":
        option = {
          tooltip: { trigger: "axis" },
          grid: { left: "25%", right: "4%", top: "8%", bottom: "8%" },
          xAxis: { type: "value", axisLabel: { fontSize: 10 } },
          yAxis: { type: "category", data: (chartData.labels || []).reverse(), axisLabel: { fontSize: 10 } },
          series: [{ type: "bar", data: (chartData.values || []).reverse(), itemStyle: { color: "#0052CC", borderRadius: [0,2,2,0] } }],
        };
        break;

      case "line":
        option = {
          tooltip: { trigger: "axis" },
          grid: { left: "8%", right: "4%", top: "8%", bottom: "8%" },
          xAxis: { type: "category", data: chartData.dates || [], axisLabel: { fontSize: 10, rotate: 30 } },
          yAxis: { type: "value", axisLabel: { fontSize: 10 } },
          series: [
            { type: "line", data: chartData.values || chartData.original || [], smooth: true, lineStyle: { color: "#0052CC", width: 2 }, areaStyle: { color: "rgba(0,82,204,0.06)" } },
          ],
        };
        if (chartData.moving_avg) {
          option.series.push({ type: "line", data: chartData.moving_avg, smooth: true, lineStyle: { color: "#F59E0B", width: 1.5, type: "dashed" } });
        }
        if (chartData.trend) {
          option.series.push({ type: "line", data: chartData.trend, smooth: true, lineStyle: { color: "#10B981", width: 1.5 } });
        }
        break;

      case "scatter":
        var points = chartData.points || [];
        // Handle clustered scatter (3-element arrays with cluster label)
        if (points.length > 0 && Array.isArray(points[0]) && points[0].length >= 3) {
          var clusters = {};
          points.forEach(function (p) {
            var c = String(p[2]);
            if (!clusters[c]) clusters[c] = [];
            clusters[c].push([p[0], p[1]]);
          });
          var scatterSeries = Object.keys(clusters).map(function (k) {
            return { type: "scatter", name: "簇" + k, data: clusters[k], symbolSize: 6 };
          });
          option = {
            tooltip: {}, grid: { left: "8%", right: "4%", top: "8%", bottom: "8%" },
            xAxis: { type: "value", axisLabel: { fontSize: 10 } },
            yAxis: { type: "value", axisLabel: { fontSize: 10 } },
            series: scatterSeries,
          };
        } else {
          option = {
            tooltip: {}, grid: { left: "8%", right: "4%", top: "8%", bottom: "8%" },
            xAxis: { type: "value", axisLabel: { fontSize: 10 } },
            yAxis: { type: "value", axisLabel: { fontSize: 10 } },
            series: [{ type: "scatter", data: points, symbolSize: 4, itemStyle: { color: "#0052CC" } }],
          };
        }
        if (chartData.trendline_x) {
          var tl = [];
          for (var i = 0; i < chartData.trendline_x.length; i++) { tl.push([chartData.trendline_x[i], chartData.trendline_y[i]]); }
          option.series.push({ type: "line", data: tl, lineStyle: { color: "#EF4444", width: 1.5 }, symbol: "none" });
        }
        break;

      case "heatmap":
        var hData = [];
        var xl = chartData.x_labels || [];
        var yl = chartData.y_labels || [];
        var mat = chartData.matrix || [];
        for (var ri = 0; ri < yl.length; ri++) {
          for (var ci = 0; ci < xl.length; ci++) { hData.push([ci, ri, mat[ri][ci]]); }
        }
        option = {
          tooltip: { position: "top" },
          grid: { left: "15%", right: "5%", top: "5%", bottom: "15%" },
          xAxis: { type: "category", data: xl, axisLabel: { fontSize: 10, rotate: 30 } },
          yAxis: { type: "category", data: yl, axisLabel: { fontSize: 10 } },
          visualMap: { min: -1, max: 1, inRange: { color: ["#3B82F6","#E5E7EB","#EF4444"] }, textStyle: { fontSize: 10 } },
          series: [{ type: "heatmap", data: hData, label: { show: true, fontSize: 9 } }],
        };
        break;

      case "boxplot":
        var bp = chartData.boxplot_series || [];
        // Render as a simplified grouped bar of medians with error bars
        var bpNames = bp.map(function (b) { return b.name; });
        var bpMedians = bp.map(function (b) { return b.median; });
        var bpMins = bp.map(function (b) { return b.min; });
        var bpMaxs = bp.map(function (b) { return b.max; });
        option = {
          tooltip: {},
          grid: { left: "8%", right: "4%", top: "8%", bottom: "8%" },
          xAxis: { type: "category", data: bpNames, axisLabel: { fontSize: 10, rotate: 20 } },
          yAxis: { type: "value", axisLabel: { fontSize: 10 } },
          series: [
            { type: "bar", data: bpMedians, itemStyle: { color: "#0052CC" } },
            { type: "scatter", data: bpMins.map(function (v, i) { return [i, v]; }), symbolSize: 6, itemStyle: { color: "#94A3B8" } },
            { type: "scatter", data: bpMaxs.map(function (v, i) { return [i, v]; }), symbolSize: 6, itemStyle: { color: "#94A3B8" } },
          ],
        };
        break;

      case "donut":
      case "pie":
        option = {
          tooltip: { trigger: "item" },
          series: [{
            type: "pie", radius: chartType === "donut" ? ["45%","70%"] : "70%", center: ["50%","50%"],
            data: (chartData.labels || []).map(function (l, i) { return { name: l, value: chartData.values[i] }; }),
            label: { fontSize: 10 },
          }],
        };
        break;

      case "qq":
        option = {
          tooltip: {}, grid: { left: "8%", right: "4%", top: "8%", bottom: "8%" },
          xAxis: { type: "value", name: "理论分位数", axisLabel: { fontSize: 10 } },
          yAxis: { type: "value", name: "样本分位数", axisLabel: { fontSize: 10 } },
          series: [
            { type: "scatter", data: (chartData.qq_x || []).map(function (x, i) { return [x, (chartData.qq_y || [])[i]]; }), symbolSize: 3, itemStyle: { color: "#0052CC" } },
          ],
        };
        break;

      case "pareto":
        var pLabels = chartData.labels || [];
        var pValues = chartData.values || [];
        var pCum = chartData.cum_pct || [];
        option = {
          tooltip: {},
          grid: { left: "8%", right: "8%", top: "8%", bottom: "8%" },
          xAxis: { type: "category", data: pLabels, axisLabel: { fontSize: 9, rotate: 30 } },
          yAxis: [
            { type: "value", name: "数值", axisLabel: { fontSize: 10 } },
            { type: "value", name: "累积%", max: 100, axisLabel: { fontSize: 10 } },
          ],
          series: [
            { type: "bar", data: pValues, itemStyle: { color: "#0052CC" } },
            { type: "line", yAxisIndex: 1, data: pCum, lineStyle: { color: "#EF4444", width: 2 }, symbol: "circle", symbolSize: 4 },
          ],
        };
        break;

      case "table":
      default:
        // Table-only result — stats and table already rendered by renderAnalysisResult, no chart needed
        if (chartType !== "table") {
          console.warn("[SSE] No chart option for type:", chartType, "data keys:", Object.keys(chartData));
        }
        chart.dispose();
        dom.innerHTML = "";
        return;
    }

    if (option) {
      try {
        chart.setOption(option);
      } catch (e) { console.error("[SSE] setOption failed:", e, chartType); }
    } else {
      console.warn("[SSE] No chart option for type:", chartType);
    }
    if (window.ResizeObserver) { new ResizeObserver(function () { chart.resize(); }).observe(dom); }
  }

  // Input events
  analysisInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendAnalysis(); }
  });
  analysisInput.addEventListener("input", function () {
    analysisInput.style.height = "auto";
    analysisInput.style.height = Math.min(analysisInput.scrollHeight, 120) + "px";
  });
  btnSendAnalysis.addEventListener("click", sendAnalysis);

  // =======================================================================
  // TAB 4: Model Training (unchanged)
  // =======================================================================
  var trainTargetCol = $("#trainTargetCol"), trainFeatureChips = $("#trainFeatureChips");
  var trainSelectError = $("#trainSelectError"), btnStartTrain = $("#btnStartTrain");
  var trainProgressArea = $("#trainProgressArea"), trainProgressPct = $("#trainProgressPct");
  var trainProgressFill = $("#trainProgressFill"), trainLogs = $("#trainLogs");
  var trainResultArea = $("#trainResultArea"), btnEnterSb = $("#btnEnterSandboxFromTrain");

  function populateTrainingSelectors() {
    if (!currentSchema) return;
    var numCols = currentSchema.columns.filter(function (c) { return c.dtype === "numeric"; });
    trainTargetCol.innerHTML = '<option value="">-- 请选择目标列（仅数值型） --</option>';
    numCols.forEach(function (c) { trainTargetCol.innerHTML += '<option value="' + escapeHtml(c.name) + '">' + escapeHtml(c.name) + '</option>'; });
    trainFeatureChips.innerHTML = "";
    numCols.forEach(function (c) {
      var chip = document.createElement("div");
      chip.className = "feature-chip"; chip.textContent = c.name; chip.dataset.column = c.name; chip.dataset.selected = "false";
      chip.addEventListener("click", function () {
        var isSel = this.dataset.selected === "true";
        this.dataset.selected = isSel ? "false" : "true"; this.classList.toggle("selected", !isSel);
      });
      trainFeatureChips.appendChild(chip);
    });
  }

  trainTargetCol.addEventListener("change", function () {
    var y = trainTargetCol.value; if (!y) return;
    $$(".feature-chip").forEach(function (chip) {
      if (chip.dataset.column === y && chip.dataset.selected === "true") { chip.dataset.selected = "false"; chip.classList.remove("selected"); }
    });
  });

  $("#btnSelectAllX").addEventListener("click", function () {
    var y = trainTargetCol.value;
    $$(".feature-chip").forEach(function (chip) {
      if (chip.dataset.column !== y) { chip.dataset.selected = "true"; chip.classList.add("selected"); }
    });
  });

  $("#btnClearAllX").addEventListener("click", function () {
    $$(".feature-chip").forEach(function (chip) { chip.dataset.selected = "false"; chip.classList.remove("selected"); });
  });

  btnStartTrain.addEventListener("click", function () {
    var y = trainTargetCol.value;
    if (!y) { trainSelectError.textContent = "请选择目标变量 (Y)。"; trainSelectError.classList.remove("hidden"); return; }
    var checked = trainFeatureChips.querySelectorAll('[data-selected="true"]');
    if (checked.length === 0) { trainSelectError.textContent = "请至少选择一个影响因素 (X)。"; trainSelectError.classList.remove("hidden"); return; }
    trainSelectError.classList.add("hidden");
    var feats = []; checked.forEach(function (chip) { feats.push(chip.dataset.column); });
    runTraining(y, feats);
  });

  function runTraining(y, feats) {
    trainProgressArea.classList.remove("hidden"); trainResultArea.classList.add("hidden"); btnEnterSb.classList.add("hidden");
    trainLogs.innerHTML = ""; trainProgressFill.style.width = "0%"; trainProgressPct.textContent = "0%";

    appendTrainLog("正在检查数据质量...");
    trainProgressFill.style.width = "5%"; trainProgressPct.textContent = "5%";

    // Step 1 — preview data quality before training
    fetch(API_BASE + "/preview-train", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ filename: currentFilename, target_col: y, feature_cols: feats }),
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (b) { throw new Error(b.detail || "预览失败: " + r.status); });
      return r.json();
    }).then(function (preview) {
      // Show preview summary
      appendTrainLog("数据总量: " + preview.total_rows.toLocaleString() + " 行");
      if (preview.bad_features && preview.bad_features.length > 0) {
        appendTrainLog("警告: " + preview.bad_features.join(", ") + " 不是数值列，无法用于训练", true);
        showTrainResultError("以下特征不是数值列，不支持用于回归分析：" + preview.bad_features.join("、") + "。请选择数值型特征（如 Quantity、UnitPrice）。");
        trainProgressFill.style.width = "100%"; trainProgressPct.textContent = "100%";
        return;
      }
      appendTrainLog("有效样本: " + preview.effective_rows.toLocaleString() + " 行 (排除 " + preview.y_dropped_pct + "% 缺失 Y)");
      // Show per-feature quality
      Object.keys(preview.features).forEach(function (f) {
        var fi = preview.features[f];
        if (fi.nan_pct > 0) {
          appendTrainLog("  " + f + ": " + fi.nan_pct + "% 缺失 (将被中位数填补)", false);
        }
      });

      // Step 2 — actual training
      appendTrainLog("正在拟合模型 + 计算诊断指标...");
      trainProgressFill.style.width = "30%"; trainProgressPct.textContent = "30%";

      return fetch(API_BASE + "/train", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ filename: currentFilename, target_col: y, feature_cols: feats }),
      });
    }).then(function (r) {
      // r is undefined if we early-returned from preview (bad features)
      if (!r) return;
      if (!r.ok) return r.json().then(function (b) { throw new Error(b.detail || "服务器返回 " + r.status); });
      return r.json();
    }).then(function (d) {
      if (!d) return;  // early return from preview
      trainProgressFill.style.width = "100%"; trainProgressPct.textContent = "100%";
      if (d.error) { appendTrainLog("失败: " + d.error, true); showTrainResultError(d.error); }
      else { appendTrainLog("训练完成", false); showTrainResult(d); }
    }).catch(function (err) {
      trainProgressFill.style.width = "100%"; trainProgressPct.textContent = "100%";
      var errMsg = err.message || "未知错误";
      // Don't show raw system errors to users
      if (errMsg.indexOf("arg must be a list") > -1 || errMsg.indexOf("1-d array") > -1) {
        errMsg = "所选特征数据格式不支持，请确保所有特征均为数值型列。";
      }
      appendTrainLog(errMsg, true); showTrainResultError(errMsg);
    });
  }

  function appendTrainLog(text, isErr) {
    var line = document.createElement("p"); line.className = "train-log-line";
    line.textContent = (isErr ? "[ERROR] " : "[OK] ") + text;
    if (isErr) line.style.color = "#EF4444";
    trainLogs.appendChild(line); trainLogs.scrollTop = trainLogs.scrollHeight;
  }

  function showTrainResultError(msg) {
    trainResultArea.classList.remove("hidden");
    var friendlyMsg = msg;
    if (msg.indexOf("无法转换为数值") > -1 || msg.indexOf("不支持用于回归") > -1) {
      friendlyMsg = msg;
    } else if (msg.indexOf("模型训练异常") > -1 || msg.indexOf("服务器返回") > -1) {
      friendlyMsg = "训练请求失败，请检查所选特征是否均为数值型列。如问题持续，请尝试更换特征组合。";
    }
    trainResultArea.innerHTML =
      '<div class="rounded-xl p-5 text-center" style="background:#FEF2F2; border:1px solid #FECACA;">' +
        '<div class="flex items-center justify-center gap-2 mb-2">' +
          '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 2a8 8 0 100 16 8 8 0 000-16zM10 6v5M10 13.5v.5" stroke="#DC2626" stroke-width="1.5" stroke-linecap="round"/></svg>' +
          '<p class="font-semibold" style="color:#991B1B;font-size:0.95rem;">模型训练未能完成</p>' +
        '</div>' +
        '<p class="text-sm mb-3" style="color:#991B1B;">' + escapeHtml(friendlyMsg) + '</p>' +
        '<p class="text-xs" style="color:#B91C1C;opacity:0.7;">建议：请确保目标变量和所有特征均为数值型列（如 Quantity、UnitPrice），避免选择文本或分类列（如 Country、Description）。</p>' +
      '</div>';
    // Also show error modal for better visibility
    showErrorModal("模型训练未能完成", friendlyMsg, "建议：请确保目标变量和所有特征均为数值型列（如 Quantity、UnitPrice），避免选择文本或分类列（如 Country、Description）。");
  }

  function showTrainResult(d) {
    var coefs = d.coefficients || {};
    var r2 = (d.r2_score != null) ? d.r2_score : 0;
    var nSamples = (d.n_samples != null) ? d.n_samples : 0;
    var diag = d.diagnostics || {};

    // Show high-cardinality warnings if any
    if (d.warnings && d.warnings.length > 0) {
      d.warnings.forEach(function (w) {
        appendTrainLog("警告: " + w, true);
      });
    }
    var residualStd = diag.residual_std || 0;
    var fi = d.feature_importance || {};

    sessionStorage.setItem("sandbox_model", JSON.stringify({
      target: d.target || "", features: d.features || [], intercept: d.intercept || 0,
      coefficients: coefs, r2_score: r2, n_samples: nSamples,
      feature_stats: d.feature_stats || {},
      residual_std: residualStd,
      target_stats: d.target_stats || {},
    }));
    var r2c = r2 >= 0.7 ? "#059669" : r2 >= 0.4 ? "#D97706" : "#EF4444";
    var r2pct = Math.round(r2 * 100);
    var cvR2 = diag.cross_val_r2 || 0;

    var coeffHtml = "";
    Object.keys(coefs).forEach(function (k) {
      var v = coefs[k], sign = v >= 0 ? "+" : "";
      var impInfo = fi[k] ? ' <span class="text-[10px]" style="color:#94A3B8;">(重要性 ' + (fi[k].relative_importance || 0).toFixed(0) + '%)</span>' : '';
      coeffHtml += '<div class="flex justify-between items-center py-1.5 border-b border-gray-50 text-xs"><span class="text-gray-600 truncate mr-2">' + escapeHtml(k) + impInfo + '</span><span class="font-mono font-medium ' + (v>=0?"text-emerald-600":"text-red-500") + '">' + sign + v.toFixed(4) + '</span></div>';
    });

    // VIF table
    var vifHtml = "";
    var vifData = diag.vif || {};
    if (Object.keys(vifData).length > 0) {
      vifHtml = '<div class="mt-3"><p class="text-xs font-semibold mb-1" style="color:#64748B;">多重共线性诊断 (VIF)</p><table class="analysis-stats-table"><thead><tr><th>特征</th><th>VIF</th><th>状态</th></tr></thead><tbody>';
      Object.keys(vifData).forEach(function (k) {
        var v = vifData[k];
        var vcls = v > 10 ? "text-red-500" : v > 5 ? "text-amber-500" : "text-emerald-600";
        var vtxt = v > 10 ? "严重" : v > 5 ? "中等" : "正常";
        vifHtml += '<tr><td>' + escapeHtml(k) + '</td><td class="font-mono">' + v.toFixed(1) + '</td><td class="' + vcls + '">' + vtxt + '</td></tr>';
      });
      vifHtml += '</tbody></table></div>';
    }
    if (diag.vif_warnings && diag.vif_warnings.length > 0) {
      vifHtml += '<div class="text-xs mt-1" style="color:#D97706;">' + diag.vif_warnings.map(escapeHtml).join("<br/>") + '</div>';
    }

    // Model comparison
    var mcHtml = "";
    var mc = diag.model_comparison || {};
    if (Object.keys(mc).length > 1) {
      mcHtml = '<div class="mt-3"><p class="text-xs font-semibold mb-1" style="color:#64748B;">模型对比（测试集）</p><table class="analysis-stats-table"><thead><tr><th>模型</th><th>R²</th><th>RMSE</th></tr></thead><tbody>';
      var best = "";
      Object.keys(mc).forEach(function (name) {
        var m = mc[name];
        if (!best || m.r2 > mc[best].r2) best = name;
        mcHtml += '<tr' + (name === "Linear" ? ' class="font-medium"' : '') + '><td>' + name + '</td><td class="font-mono">' + m.r2.toFixed(4) + '</td><td class="font-mono">' + m.rmse.toFixed(4) + '</td></tr>';
      });
      mcHtml += '</tbody></table></div>';
      if (best !== "Linear") {
        mcHtml += '<p class="text-xs mt-1" style="color:#059669;">' + best + ' 在测试集上表现优于普通线性回归。</p>';
      }
    }

    // Residual charts (rendered after DOM insertion)
    var chartContainerId = "residualCharts-" + Date.now();

    trainResultArea.classList.remove("hidden");
    trainResultArea.innerHTML =
      '<div class="grid grid-cols-2 gap-3 mb-4">' +
        '<div class="text-center"><p class="text-xs text-gray-400 uppercase tracking-wider mb-1">R²</p>' +
          '<div class="r2-gauge mx-auto"><svg width="48" height="48" viewBox="0 0 64 64"><circle class="r2-bg" cx="32" cy="32" r="28"/><circle class="r2-fill" cx="32" cy="32" r="28" stroke="' + r2c + '" stroke-dasharray="' + (Math.PI*56) + '" stroke-dashoffset="' + (Math.PI*56*(1-r2)) + '"/></svg><span class="r2-text" style="font-size:0.75rem;color:' + r2c + '">' + r2pct + '%</span></div></div>' +
        '<div class="text-center"><p class="text-xs text-gray-400 uppercase tracking-wider mb-1">5-fold CV R²</p><p class="text-xl font-bold" style="color:#1E293B;">' + (cvR2 * 100).toFixed(1) + '%</p><p class="text-[10px]" style="color:#94A3B8;">±' + ((diag.cross_val_r2_std || 0) * 100).toFixed(1) + '%</p></div>' +
      '</div>' +
      '<p class="text-xs text-gray-500 text-center mb-3">有效数据 ' + nSamples.toLocaleString() + ' 条' +
        (d.n_dropped > 0 ? ' | 剔除 ' + d.n_dropped + ' 条 Y 缺失' : '') +
        (d.n_imputed > 0 ? ' | 填补 ' + d.n_imputed + ' 个缺失值' : '') +
      '</p>' +
      '<div id="' + chartContainerId + '_pv" class="chart-box mb-3" style="min-height:220px;"></div>' +
      '<div id="' + chartContainerId + '_qq" class="chart-box mb-3" style="min-height:220px;"></div>' +
      vifHtml +
      mcHtml +
      '<div class="rounded-xl p-3 mt-3" style="background:#F8FAFC; border:1px solid #E2E8F0;"><p class="text-xs font-medium mb-2" style="color:#64748B;">回归方程: Y = ' + (d.intercept || 0).toFixed(4) + ' + Σ(βi x Xi)</p>' + coeffHtml + '</div>';
    btnEnterSb.classList.remove("hidden");

    // Render residual diagnostic charts
    setTimeout(function () {
      renderResidualCharts(chartContainerId, diag);
    }, 100);
  }

  function renderResidualCharts(baseId, diag) {
    if (typeof echarts === "undefined") return;
    var residuals = diag.residuals || {};

    // Predicted vs Actual scatter
    var pvDom = $("#" + baseId + "_pv");
    if (pvDom && residuals.pred_vs_actual) {
      pvDom.innerHTML = "";
      var pvChart = echarts.init(pvDom);
      var pvData = residuals.pred_vs_actual;
      // Compute min/max for diagonal line
      var allVals = []; pvData.forEach(function (p) { allVals.push(p[0], p[1]); });
      var pvMin = Math.min.apply(null, allVals); var pvMax = Math.max.apply(null, allVals);
      pvChart.setOption({
        title: { text: "预测值 vs 实际值", left: "center", textStyle: { fontSize: 12, color: "#64748B" } },
        tooltip: {}, grid: { left: "10%", right: "4%", top: "15%", bottom: "8%" },
        xAxis: { type: "value", name: "实际值", axisLabel: { fontSize: 10 } },
        yAxis: { type: "value", name: "预测值", axisLabel: { fontSize: 10 } },
        series: [
          { type: "scatter", data: pvData, symbolSize: 4, itemStyle: { color: "#0052CC", opacity: 0.5 } },
          { type: "line", data: [[pvMin, pvMin], [pvMax, pvMax]], lineStyle: { color: "#EF4444", type: "dashed", width: 1 }, symbol: "none" },
        ],
      });
      if (window.ResizeObserver) { new ResizeObserver(function () { pvChart.resize(); }).observe(pvDom); }
    }

    // Residual QQ plot
    var qqDom = $("#" + baseId + "_qq");
    if (qqDom && residuals.qq_x) {
      qqDom.innerHTML = "";
      var qqChart = echarts.init(qqDom);
      var qqPairs = residuals.qq_x.map(function (x, i) { return [x, residuals.qq_y[i]]; });
      var qqExtent = Math.max(Math.abs(residuals.qq_x[residuals.qq_x.length - 1] || 3), Math.abs(residuals.qq_y[residuals.qq_y.length - 1] || 3));
      qqChart.setOption({
        title: { text: "残差 Q-Q 图", left: "center", textStyle: { fontSize: 12, color: "#64748B" } },
        tooltip: {}, grid: { left: "10%", right: "4%", top: "15%", bottom: "8%" },
        xAxis: { type: "value", name: "理论分位数", axisLabel: { fontSize: 10 } },
        yAxis: { type: "value", name: "样本分位数", axisLabel: { fontSize: 10 } },
        series: [
          { type: "scatter", data: qqPairs, symbolSize: 4, itemStyle: { color: "#0052CC", opacity: 0.5 } },
          { type: "line", data: [[-qqExtent, -qqExtent], [qqExtent, qqExtent]], lineStyle: { color: "#EF4444", type: "dashed", width: 1 }, symbol: "none" },
        ],
      });
      if (window.ResizeObserver) { new ResizeObserver(function () { qqChart.resize(); }).observe(qqDom); }
    }
  }

  btnEnterSb.addEventListener("click", function () { switchTab("sandbox"); });

  // =======================================================================
  // TAB 5: What-If Simulation Sandbox (enhanced)
  // =======================================================================
  var sandboxModel = null;
  var sandboxSliderValues = {};
  var sandboxWaterfallChart = null;
  var sandboxScenarioChart = null;
  var sandboxSensChart = null;
  var sandboxRafId = null;
  var sandboxEmpty = $("#sandboxEmpty"), sandboxActive = $("#sandboxActive");
  var sandboxSliders = $("#sandboxSliders");
  var sandboxPredEl = $("#sandboxPrediction"), sandboxTargetLabel = $("#sandboxTargetLabel");
  var sandboxFormulaEl = $("#sandboxFormula");
  var sandboxWaterfallDom = $("#sandboxWaterfall");
  var sandboxScenarioDom = $("#sandboxScenarioChart");
  var sandboxStatusEl = $("#sandboxStatus");
  var sandboxScenarios = [];     // [{name, values, prediction}]
  var sandboxResidualStd = 0;    // from training diagnostics
  var sandboxTargetStats = null; // {mean, std} from training

  function computePrediction() {
    if (!sandboxModel) return { y: 0, lower: 0, upper: 0 };
    var y = sandboxModel.intercept;
    var feats = sandboxModel.features;
    var coefs = sandboxModel.coefficients;
    for (var i = 0; i < feats.length; i++) {
      y += (sandboxSliderValues[feats[i]] || 0) * (coefs[feats[i]] || 0);
    }
    var half = 1.96 * sandboxResidualStd;
    return { y: y, lower: y - half, upper: y + half };
  }

  var countUpAnimId = null, countUpCurrent = 0;
  function countUpTo(target) {
    if (countUpAnimId) cancelAnimationFrame(countUpAnimId);
    var from = countUpCurrent, to = target, duration = 400, start = null;
    function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
    function step(ts) {
      if (!start) start = ts;
      var elapsed = ts - start, progress = Math.min(elapsed / duration, 1);
      var eased = easeOutCubic(progress), current = from + (to - from) * eased;
      countUpCurrent = current; sandboxPredEl.textContent = current.toFixed(2);
      if (progress < 1) { countUpAnimId = requestAnimationFrame(step); }
      else { countUpAnimId = null; countUpCurrent = to; sandboxPredEl.textContent = to.toFixed(2); }
    }
    countUpAnimId = requestAnimationFrame(step);
  }

  function updatePredictionDisplay() {
    var p = computePrediction();
    countUpTo(p.y);
    scheduleChartUpdate(p);
  }
  function scheduleChartUpdate(p) {
    if (sandboxRafId) cancelAnimationFrame(sandboxRafId);
    sandboxRafId = requestAnimationFrame(function () {
      sandboxRafId = null;
      updateWaterfallChart(p);
      updateScenarioChart();
    });
  }

  // ---- Waterfall: contribution breakdown ----
  function initWaterfallChart(pred) {
    if (sandboxWaterfallChart) { sandboxWaterfallChart.dispose(); sandboxWaterfallChart = null; }
    sandboxWaterfallDom.innerHTML = "";
    if (typeof echarts === "undefined") return;
    sandboxWaterfallChart = echarts.init(sandboxWaterfallDom);
    updateWaterfallChart(pred);
  }

  function updateWaterfallChart(pred) {
    if (!sandboxWaterfallChart || sandboxWaterfallChart.isDisposed()) return;
    if (!sandboxModel) return;
    var feats = sandboxModel.features;
    var coefs = sandboxModel.coefficients;
    var total = pred ? pred.y : 0;
    var interval = sandboxResidualStd > 0 ? " ±" + (sandboxResidualStd * 1.96).toFixed(2) : "";

    // Build contribution data
    var labels = [];
    var values = [];
    var colors = [];
    var intercept = sandboxModel.intercept || 0;

    feats.forEach(function (f) {
      var v = (sandboxSliderValues[f] || 0) * (coefs[f] || 0);
      labels.push(f);
      values.push(parseFloat(v.toFixed(3)));
      colors.push(v >= 0 ? "#059669" : "#EF4444");
    });

    sandboxWaterfallChart.setOption({
      tooltip: { trigger: "axis", formatter: function (p) { return p[0].name + ": " + p[0].value.toFixed(3); } },
      grid: { left: "30%", right: "8%", top: "4%", bottom: "4%" },
      xAxis: {
        type: "value",
        axisLabel: { fontSize: 10 },
        splitLine: { lineStyle: { color: "#F1F5F9" } },
      },
      yAxis: {
        type: "category",
        data: labels,
        axisLabel: { fontSize: 11, color: "#334155", width: 120, overflow: "truncate" },
        axisLine: { show: false }, axisTick: { show: false },
      },
      series: [{
        type: "bar",
        data: values.map(function (v, i) {
          return {
            value: v,
            itemStyle: {
              color: colors[i],
              borderRadius: v >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3],
            },
          };
        }),
        barWidth: 18,
        label: { show: true, position: "right", fontSize: 9, color: "#64748B",
          formatter: function (p) { return p.value.toFixed(2); } },
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: "#1E293B", type: "dashed", width: 1 },
          data: [{ xAxis: total, label: { formatter: "预测: " + total.toFixed(2) + interval, fontSize: 10, color: "#0052CC", fontWeight: "bold" } }],
        },
      }],
    });
  }

  // ---- Scenario comparison chart ----
  function initScenarioChart() {
    if (sandboxScenarioChart) { sandboxScenarioChart.dispose(); sandboxScenarioChart = null; }
    sandboxScenarioDom.innerHTML = "";
    if (typeof echarts === "undefined") return;
    sandboxScenarioChart = echarts.init(sandboxScenarioDom);
    updateScenarioChart();
  }

  function updateScenarioChart() {
    if (!sandboxScenarioChart || sandboxScenarioChart.isDisposed()) return;
    var current = computePrediction();
    var labels = ["当前"];
    var values = [parseFloat(current.y.toFixed(2))];
    var errors = sandboxResidualStd > 0 ? [[parseFloat(current.lower.toFixed(2)), parseFloat(current.upper.toFixed(2))]] : [[current.y, current.y]];

    sandboxScenarios.forEach(function (s) {
      labels.push(s.name);
      values.push(parseFloat(s.prediction.toFixed(2)));
      errors.push([s.prediction, s.prediction]);  // scenarios don't have their own residual std
    });

    sandboxScenarioChart.setOption({
      tooltip: { trigger: "axis" },
      grid: { left: "8%", right: "4%", top: "8%", bottom: "8%" },
      xAxis: { type: "category", data: labels, axisLabel: { fontSize: 10 } },
      yAxis: { type: "value", axisLabel: { fontSize: 10 } },
      series: [{
        type: "bar",
        data: values.map(function (v, i) { return { value: v, itemStyle: { color: i === 0 ? "#0052CC" : "#94A3B8", borderRadius: [3, 3, 0, 0] } }; }),
        barWidth: "40%",
        label: { show: true, position: "top", fontSize: 10, color: "#1E293B", formatter: function (p) { return p.value.toFixed(2); } },
        markLine: sandboxResidualStd > 0 ? {
          silent: true, symbol: "none",
          lineStyle: { color: "#F56C6C", type: "dashed", width: 1 },
          data: [{ yAxis: current.upper, label: { formatter: "上限", fontSize: 9, color: "#F56C6C" } },
                 { yAxis: current.lower, label: { formatter: "下限", fontSize: 9, color: "#F56C6C" } }],
        } : undefined,
      }],
    });
  }

  function buildSliderCards() {
    if (!sandboxModel) return;
    var feats = sandboxModel.features, coefs = sandboxModel.coefficients, stats = sandboxModel.feature_stats || {};
    var html = "";
    for (var i = 0; i < feats.length; i++) {
      var f = feats[i], coef = coefs[f] || 0, st = stats[f] || {};
      var minVal = (st.min != null) ? st.min : -Math.abs(coef) * 100;
      var maxVal = (st.max != null) ? st.max : Math.abs(coef) * 100;
      var meanVal = (st.mean != null) ? st.mean : 0;
      var range = maxVal - minVal || 1, step = range / 100;
      if (step < 0.001) step = 0.001; else if (step < 0.01) step = Math.round(step * 10000) / 10000; else if (step < 1) step = Math.round(step * 100) / 100; else step = Math.round(step * 10) / 10;
      sandboxSliderValues[f] = meanVal;
      html += '<div class="card" style="padding:1rem 1.25rem;">' +
        '<div class="flex items-center justify-between mb-2"><span class="text-sm font-semibold truncate mr-2" style="color:#1E293B;">' + escapeHtml(f) + '</span><span class="text-xs font-mono" style="color:' + (coef >= 0 ? "#059669" : "#EF4444") + ';">' + (coef >= 0 ? "+" : "") + coef.toFixed(4) + '</span></div>' +
        '<div class="flex items-center gap-3">' +
          '<input type="range" class="sandbox-slider" data-feature="' + escapeHtml(f) + '" min="' + minVal + '" max="' + maxVal + '" step="' + step + '" value="' + meanVal + '" style="flex:1; height:6px; -webkit-appearance:none; appearance:none; background:#E2E8F0; border-radius:3px; outline:none; cursor:pointer;" />' +
          '<input type="number" class="sandbox-number" data-feature="' + escapeHtml(f) + '" value="' + meanVal + '" step="' + step + '" min="' + minVal + '" max="' + maxVal + '" style="width:5.5rem; padding:0.375rem 0.5rem; font-size:0.8125rem; text-align:center; border:1px solid #E2E8F0; border-radius:0.5rem; color:#1E293B; font-variant-numeric:tabular-nums; outline:none;" /></div></div>';
    }
    sandboxSliders.innerHTML = html;
    sandboxSliders.querySelectorAll(".sandbox-slider").forEach(function (slider) {
      slider.addEventListener("input", function () {
        var feat = this.dataset.feature, val = parseFloat(this.value);
        sandboxSliderValues[feat] = val;
        var numInput = sandboxSliders.querySelector('.sandbox-number[data-feature="' + feat + '"]');
        if (numInput) numInput.value = val; updatePredictionDisplay();
      });
    });
    sandboxSliders.querySelectorAll(".sandbox-number").forEach(function (num) {
      num.addEventListener("input", function () {
        var feat = this.dataset.feature, val = parseFloat(this.value) || 0;
        sandboxSliderValues[feat] = val;
        var slider = sandboxSliders.querySelector('.sandbox-slider[data-feature="' + feat + '"]');
        if (slider) slider.value = val; updatePredictionDisplay();
      });
    });
  }

  function initSandbox() {
    if (!currentSchema) { sandboxEmpty.classList.remove("hidden"); sandboxActive.classList.add("hidden"); sandboxStatusEl.textContent = "请先在「数据接入」上传表格，再前往「模型训练」完成模型拟合。"; return; }
    var raw = sessionStorage.getItem("sandbox_model");
    if (!raw) { sandboxEmpty.classList.remove("hidden"); sandboxActive.classList.add("hidden"); sandboxStatusEl.textContent = "尚未训练模型 — 请在「模型训练」Tab 中完成训练后解锁。"; return; }
    sandboxModel = JSON.parse(raw); sandboxSliderValues = {};
    sandboxResidualStd = sandboxModel.residual_std || 0;
    sandboxTargetStats = sandboxModel.target_stats || null;
    sandboxScenarios = [];
    if (sandboxWaterfallChart) { sandboxWaterfallChart.dispose(); sandboxWaterfallChart = null; }
    if (sandboxScenarioChart) { sandboxScenarioChart.dispose(); sandboxScenarioChart = null; }
    if (sandboxSensChart) { sandboxSensChart.dispose(); sandboxSensChart = null; }
    sandboxEmpty.classList.add("hidden"); sandboxActive.classList.remove("hidden");
    sandboxTargetLabel.textContent = "目标：" + sandboxModel.target + " | R² = " + (sandboxModel.r2_score * 100).toFixed(1) + "%" + (sandboxResidualStd > 0 ? " | 预测区间 ±" + (sandboxResidualStd * 1.96).toFixed(2) : "");
    sandboxFormulaEl.textContent = "Y = " + sandboxModel.intercept.toFixed(4) + sandboxModel.features.map(function (f) { var c = sandboxModel.coefficients[f] || 0; return (c >= 0 ? " + " : " - ") + Math.abs(c).toFixed(4) + " x " + f; }).join("");
    buildSliderCards();
    renderScenarioBar();
    renderSensitivityChart();
    renderOptimizerBtn();
    var pred = computePrediction();
    initWaterfallChart(pred);
    initScenarioChart();
    countUpCurrent = pred.y; sandboxPredEl.textContent = pred.y.toFixed(2); scheduleChartUpdate(pred);
  }

  // ---- Scenario management ----
  function renderScenarioBar() {
    var existing = $("#sandboxScenarioBar");
    if (existing) existing.remove();
    var bar = document.createElement("div");
    bar.id = "sandboxScenarioBar";
    bar.className = "flex items-center gap-2 mb-3";
    bar.innerHTML = '<span class="text-xs font-semibold" style="color:#64748B;">场景对比</span>' +
      '<div id="sandboxScenarioChips" class="flex gap-1.5"></div>' +
      '<button id="btnSaveScenario" class="text-xs font-medium px-2 py-0.5 rounded border" style="color:#0052CC; border-color:#0052CC; background:#fff;">保存当前</button>';
    sandboxSliders.insertBefore(bar, sandboxSliders.firstChild);
    updateScenarioChips();
    $("#btnSaveScenario").addEventListener("click", function () {
      if (sandboxScenarios.length >= 3) { showToast("最多保存 3 个场景"); return; }
      var vals = {}; Object.keys(sandboxSliderValues).forEach(function (k) { vals[k] = sandboxSliderValues[k]; });
      var pred = computePrediction();
      sandboxScenarios.push({ name: "场景" + (sandboxScenarios.length + 1), values: vals, prediction: pred.y });
      updateScenarioChips();
      updateScenarioChart();
    });
  }

  function updateScenarioChips() {
    var chips = $("#sandboxScenarioChips"); if (!chips) return;
    var html = "";
    sandboxScenarios.forEach(function (s, i) {
      html += '<span class="scenario-chip" data-scenario-idx="' + i + '" style="cursor:pointer; padding:0.125rem 0.5rem; border-radius:999px; font-size:0.6875rem; border:1px solid #E2E8F0; color:#64748B; background:#fff;">' + escapeHtml(s.name) + ' (' + s.prediction.toFixed(0) + ')</span>';
    });
    chips.innerHTML = html;
    chips.querySelectorAll(".scenario-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        var idx = parseInt(this.dataset.scenarioIdx);
        var s = sandboxScenarios[idx]; if (!s) return;
        Object.keys(s.values).forEach(function (k) { sandboxSliderValues[k] = s.values[k]; });
        // Update UI
        $$(".sandbox-slider").forEach(function (sl) { if (s.values[sl.dataset.feature] != null) sl.value = s.values[sl.dataset.feature]; });
        $$(".sandbox-number").forEach(function (n) { if (s.values[n.dataset.feature] != null) n.value = s.values[n.dataset.feature]; });
        updatePredictionDisplay();
      });
    });
  }

  // ---- Sensitivity Analysis ----
  function renderSensitivityChart() {
    var existing = $("#sandboxSensChart");
    if (existing) existing.remove();
    if (!sandboxModel) return;
    var container = document.createElement("div");
    container.id = "sandboxSensChart";
    container.className = "card mt-3";
    container.style.cssText = "padding:1rem 1.25rem;";
    container.innerHTML = '<h4 class="text-xs font-semibold mb-2" style="color:#64748B;">敏感性分析</h4><div id="sandboxSensChartInner" style="height:140px;"></div>';
    // Insert after the reset button area
    var slidersContainer = sandboxSliders;
    slidersContainer.appendChild(container);

    setTimeout(function () {
      if (typeof echarts === "undefined") return;
      var feats = sandboxModel.features;
      var coefs = sandboxModel.coefficients;
      var stats = sandboxModel.feature_stats || {};
      var impacts = [];
      feats.forEach(function (f) {
        var st = stats[f] || {};
        var minV = st.min != null ? st.min : 0;
        var maxV = st.max != null ? st.max : 0;
        var coef = coefs[f] || 0;
        var impact = coef * (maxV - minV);
        impacts.push({ name: f, impact: Math.abs(impact), coef: coef });
      });
      impacts.sort(function (a, b) { return b.impact - a.impact; });

      sandboxSensChart = echarts.init($("#sandboxSensChartInner"));
      sandboxSensChart.setOption({
        tooltip: { trigger: "axis", formatter: function (params) {
          var d = params[0]; return d.name + "<br/>影响幅度: " + Math.abs(d.value).toFixed(2);
        } },
        grid: { left: "25%", right: "4%", top: "4%", bottom: "4%" },
        xAxis: { type: "value", axisLabel: { fontSize: 10 } },
        yAxis: { type: "category", data: impacts.map(function (imp) { return imp.name; }).reverse(), axisLabel: { fontSize: 10 } },
        series: [{
          type: "bar", data: impacts.map(function (imp) { return imp.impact; }).reverse(),
          itemStyle: { color: "#0052CC", borderRadius: [0, 2, 2, 0] },
          label: { show: true, position: "right", fontSize: 9, formatter: function (p) { return p.value.toFixed(2); } },
        }],
      });
    }, 200);
  }

  // ---- Auto-optimizer ----
  function renderOptimizerBtn() {
    var existing = $("#sandboxOptimizerArea");
    if (existing) existing.remove();
    var area = document.createElement("div");
    area.id = "sandboxOptimizerArea";
    area.className = "card mt-3";
    area.style.cssText = "padding:1rem 1.25rem;";
    area.innerHTML =
      '<div class="flex items-center justify-between">' +
        '<div><h4 class="text-xs font-semibold" style="color:#64748B;">自动寻优</h4><p class="text-[10px]" style="color:#94A3B8;">在滑块范围内寻找最大化或最小化目标的最优参数组合</p></div>' +
        '<div class="flex gap-1.5">' +
          '<button id="btnOptimizeMax" class="text-xs font-medium px-2.5 py-1 rounded text-white hover:opacity-90 transition-all" style="background:#059669;">最大化 Y</button>' +
          '<button id="btnOptimizeMin" class="text-xs font-medium px-2.5 py-1 rounded text-white hover:opacity-90 transition-all" style="background:#EF4444;">最小化 Y</button>' +
        '</div>' +
      '</div>';
    sandboxSliders.appendChild(area);

    $("#btnOptimizeMax").addEventListener("click", function () { runOptimizer("max"); });
    $("#btnOptimizeMin").addEventListener("click", function () { runOptimizer("min"); });
  }

  function runOptimizer(direction) {
    if (!sandboxModel) return;
    var feats = sandboxModel.features;
    var coefs = sandboxModel.coefficients;
    var stats = sandboxModel.feature_stats || {};
    var bestVals = {};
    feats.forEach(function (f) {
      var st = stats[f] || {};
      var minV = st.min != null ? st.min : -100;
      var maxV = st.max != null ? st.max : 100;
      var coef = coefs[f] || 0;
      if (direction === "max") {
        bestVals[f] = coef > 0 ? maxV : minV;
      } else {
        bestVals[f] = coef > 0 ? minV : maxV;
      }
      sandboxSliderValues[f] = bestVals[f];
    });
    // Update all sliders
    $$(".sandbox-slider").forEach(function (sl) { if (bestVals[sl.dataset.feature] != null) sl.value = bestVals[sl.dataset.feature]; });
    $$(".sandbox-number").forEach(function (n) { if (bestVals[n.dataset.feature] != null) n.value = bestVals[n.dataset.feature]; });
    updatePredictionDisplay();
    showToast(direction === "max" ? "已调整到最大化 Y 的参数组合" : "已调整到最小化 Y 的参数组合", 2000);
  }

  var btnResetSliders = $("#btnResetSliders");
  if (btnResetSliders) {
    btnResetSliders.addEventListener("click", function () {
      if (!sandboxModel) return;
      var stats = sandboxModel.feature_stats || {};
      $$(".sandbox-slider").forEach(function (slider) {
        var feat = slider.dataset.feature, st = stats[feat] || {}, meanVal = (st.mean != null) ? st.mean : 0;
        slider.value = meanVal; sandboxSliderValues[feat] = meanVal;
      });
      $$(".sandbox-number").forEach(function (num) {
        var feat = num.dataset.feature, st = stats[feat] || {}, meanVal = (st.mean != null) ? st.mean : 0;
        num.value = meanVal;
      });
      updatePredictionDisplay();
    });
  }

  function updateSandboxStatus() {
    if (!currentSchema) { sandboxEmpty.classList.remove("hidden"); sandboxActive.classList.add("hidden"); sandboxStatusEl.textContent = "请先在「数据接入」上传表格，再前往「模型训练」完成模型拟合。"; return; }
    var raw = sessionStorage.getItem("sandbox_model");
    if (raw) {
      sandboxModel = JSON.parse(raw);
      sandboxStatusEl.innerHTML = '<span class="text-green-600 font-medium">模型已就绪</span> — 目标: ' + escapeHtml(sandboxModel.target) + ' | R²: ' + (sandboxModel.r2_score * 100).toFixed(1) + '% | 特征: ' + sandboxModel.features.join(", ");
      initSandbox();
    } else { sandboxEmpty.classList.remove("hidden"); sandboxActive.classList.add("hidden"); sandboxStatusEl.textContent = "尚未训练模型 — 请在「模型训练」Tab 中完成训练后解锁。"; }
  }

  window.addEventListener("resize", function () {
    if (sandboxWaterfallChart && !sandboxWaterfallChart.isDisposed()) sandboxWaterfallChart.resize();
    if (sandboxScenarioChart && !sandboxScenarioChart.isDisposed()) sandboxScenarioChart.resize();
  });

  // =======================================================================
  // Init
  // =======================================================================
  function init() {
    loadSettingsToInputs();
    loadAuthState();
    console.log("[SmartAnalysis] Refactored architecture ready. API: " + API_BASE);
  }
  init();
})();
