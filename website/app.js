/* ═══════════════════════════════════════════════════════════════════
   app.js — Shared state, API client, nav helpers
   MedRecords RAG · used by index.html, auth.html, try.html
═══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  /* ── Storage key ── */
  var STORAGE_KEY = 'medrag.state';

  /* ── Initialise state immediately (before DOMContentLoaded) ── */
  var _initialState = {
    access_token: null,
    refresh_token: null,
    username: null,
    role: null,
    exp: null,
  };

  try {
    var _raw = localStorage.getItem(STORAGE_KEY);
    if (_raw) {
      var _parsed = JSON.parse(_raw);
      if (_parsed && typeof _parsed === 'object') {
        _initialState = Object.assign(_initialState, _parsed);
      }
    }
  } catch (_) { /* localStorage may be unavailable in private browsing */ }

  /* ══════════════════════════════════════════════════════════════
     PUBLIC NAMESPACE
  ══════════════════════════════════════════════════════════════ */
  window.MedRag = {

    API_BASE: 'http://localhost:8000',

    /* ── Live state object — hydrated from localStorage above ── */
    state: _initialState,

    /* ──────────────────────────────────────────────────────────
       STATE PERSISTENCE
    ────────────────────────────────────────────────────────── */
    loadState: function () {
      try {
        var raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
          var parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object') {
            this.state = Object.assign({
              access_token: null,
              refresh_token: null,
              username: null,
              role: null,
              exp: null,
            }, parsed);
          }
        }
      } catch (_) {}
    },

    saveState: function () {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state));
      } catch (_) {}
    },

    clearState: function () {
      this.state = {
        access_token: null,
        refresh_token: null,
        username: null,
        role: null,
        exp: null,
      };
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (_) {}
    },

    /* ──────────────────────────────────────────────────────────
       JWT DECODE — no external library
       Returns the payload object or null on failure.
    ────────────────────────────────────────────────────────── */
    decodeJwt: function (token) {
      try {
        var middle = token.split('.')[1];
        if (!middle) return null;
        var padded = middle + '==='.slice((middle.length + 3) % 4);
        var json = atob(padded.replace(/-/g, '+').replace(/_/g, '/'));
        return JSON.parse(json);
      } catch (_) {
        return null;
      }
    },

    /* ──────────────────────────────────────────────────────────
       API CLIENT
       Returns { status, headers, body }
       Automatically attaches Authorization: Bearer if logged in.
    ────────────────────────────────────────────────────────── */
    call: async function (path, opts) {
      opts = opts || {};
      var headers = opts.headers ? Object.assign({}, opts.headers) : {};

      if (this.state.access_token && !headers['Authorization']) {
        headers['Authorization'] = 'Bearer ' + this.state.access_token;
      }

      var fetchOpts = Object.assign({}, opts, { headers: headers });
      var resp = await fetch(this.API_BASE + path, fetchOpts);
      var text = await resp.text();
      var json = null;
      try { json = JSON.parse(text); } catch (_) { /* not JSON */ }

      return {
        status: resp.status,
        headers: resp.headers,
        body: json !== null ? json : text,
      };
    },

    /* ──────────────────────────────────────────────────────────
       DISPLAY RESPONSE
       Renders a { status, body } result into a DOM element.
    ────────────────────────────────────────────────────────── */
    displayResponse: function (el, result) {
      el.classList.remove('ok', 'err');
      el.classList.add(result.status >= 200 && result.status < 300 ? 'ok' : 'err');

      var body = typeof result.body === 'string'
        ? result.body
        : JSON.stringify(result.body, null, 2);

      el.textContent = 'HTTP ' + result.status + '\n\n' + body;
    },

    /* ──────────────────────────────────────────────────────────
       INIT NAV
       Renders the auth pill on the nav and wires the logout
       button if one is present on the current page.
    ────────────────────────────────────────────────────────── */
    initNav: function () {
      var self = this;

      /* Scroll border + backdrop-filter blur */
      var nav = document.getElementById('mainNav');
      if (nav) {
        window.addEventListener('scroll', function () {
          if (window.scrollY > 40) {
            nav.classList.add('scrolled');
          } else {
            nav.classList.remove('scrolled');
          }
        }, { passive: true });
      }

      /* Hamburger toggle */
      var hamburger = document.getElementById('hamburgerBtn');
      var mobileMenu = document.getElementById('mobileMenu');
      if (hamburger && mobileMenu) {
        hamburger.addEventListener('click', function () {
          var isOpen = mobileMenu.classList.contains('open');
          mobileMenu.classList.toggle('open', !isOpen);
          hamburger.setAttribute('aria-expanded', String(!isOpen));
        });

        var mobileLinks = mobileMenu.querySelectorAll('a');
        mobileLinks.forEach(function (link) {
          link.addEventListener('click', function () {
            mobileMenu.classList.remove('open');
            hamburger.setAttribute('aria-expanded', 'false');
          });
        });
      }

      /* Update the auth pill */
      self._updateNavPill();

      /* Auto-refresh pill every 60 s */
      setInterval(function () {
        self._updateNavPill();
      }, 60000);

      /* Wire any logout button on the nav */
      var navLogoutBtn = document.getElementById('navLogoutBtn');
      if (navLogoutBtn) {
        navLogoutBtn.addEventListener('click', async function () {
          if (self.state.access_token) {
            try {
              await self.call('/auth/logout', { method: 'POST' });
            } catch (_) { /* ignore network errors on logout */ }
          }
          self.clearState();
          self._updateNavPill();
          /* If we're on a protected page, redirect to auth */
          if (window._medragProtectedPage) {
            window.location.href = 'auth.html';
          }
        });
      }
    },

    /* Internal: update [data-nav-status] elements on page */
    _updateNavPill: function () {
      var pillEls = document.querySelectorAll('[data-nav-status]');
      pillEls.forEach(function (el) {
        if (!window.MedRag.state.access_token) {
          el.textContent = '';
          el.className = el.className.replace(/\bnav-auth-pill\b/, '').trim();
        } else {
          var mins = 0;
          if (window.MedRag.state.exp) {
            mins = Math.max(0, Math.round(
              (window.MedRag.state.exp * 1000 - Date.now()) / 60000
            ));
          }
          el.textContent = '● ' +
            window.MedRag.state.username +
            ' (' + window.MedRag.state.role + ')' +
            ' · ' + mins + 'm';
          if (!el.classList.contains('nav-auth-pill')) {
            el.classList.add('nav-auth-pill');
          }
        }
      });

      /* Also update the inline status bars used on auth/try pages */
      var statusBars = document.querySelectorAll('.status-bar[data-nav-status]');
      statusBars.forEach(function (bar) {
        if (!window.MedRag.state.access_token) {
          bar.classList.remove('in');
          bar.innerHTML = '<span>● not logged in</span>';
        } else {
          bar.classList.add('in');
          var mins2 = 0;
          if (window.MedRag.state.exp) {
            mins2 = Math.max(0, Math.round(
              (window.MedRag.state.exp * 1000 - Date.now()) / 60000
            ));
          }
          bar.innerHTML =
            '<span>● logged in: ' +
            window.MedRag.state.username +
            ' (' + window.MedRag.state.role + ')' +
            ' · access expires in ' + mins2 + 'min</span>';
        }
      });
    },

    /* ──────────────────────────────────────────────────────────
       REQUIRE AUTH
       Call on protected pages. Redirects if no token present.
    ────────────────────────────────────────────────────────── */
    requireAuth: function (redirectTo) {
      redirectTo = redirectTo || 'auth.html';
      if (!this.state.access_token) {
        window.location.href = redirectTo;
      }
    },

  }; /* end window.MedRag */

  /* ── Auto-run initNav on DOMContentLoaded ── */
  document.addEventListener('DOMContentLoaded', function () {
    window.MedRag.initNav();
  });

})();
