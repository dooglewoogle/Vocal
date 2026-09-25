# VocalSpike: does a tiny app bundle give Vocal its own macOS permissions?

Throwaway experiment. Today macOS gives Vocal's permissions to whatever terminal starts it. This checks whether wrapping Vocal in a small `.app` fixes that, before we build the real thing.

Everything the probe learns goes to `~/Library/Logs/VocalSpike.log`. **Send that file back at the end.** The probe speaks instructions ("type some keys…") out loud.

## Setup

```bash
cd ~/Downloads/Vocal                     # the Vocal checkout
scripts/macos_launcher_spike/build.sh    # builds ~/Applications/VocalSpike.app
tccutil reset All dev.vocal.spike        # start from a clean slate
```

## Steps

Write down what each macOS prompt says, especially **which app name** it asks about.

| # | Run | Then |
|---|-----|------|
| 1 | `~/Applications/VocalSpike.app/Contents/MacOS/VocalSpike spawn` | nothing; this runs it directly from iTerm |
| 2 | `open ~/Applications/VocalSpike.app --args spawn` | if prompted, or in System Settings → Privacy & Security, switch **VocalSpike** on in Input Monitoring and Accessibility |
| 3 | `open ~/Applications/VocalSpike.app --args spawn` | run again after granting |
| 4 | `open ~/Applications/VocalSpike.app --args exec` | |
| 5 | `open ~/Applications/VocalSpike.app --args disclaim` | |
| 6 | `scripts/macos_launcher_spike/build.sh` then `open ~/Applications/VocalSpike.app --args spawn` | don't touch Settings: does the grant survive a rebuild? Does Settings still show VocalSpike as on? |
| 7 | `tccutil reset All dev.vocal.spike`, grant VocalSpike **only** Accessibility, then `open ~/Applications/VocalSpike.app --args spawn` | |
| 8 | Open TextEdit with an empty document, then `open ~/Applications/VocalSpike.app --args spawn paste` | click into TextEdit when told to. Write down exactly what appeared. Then repeat while holding Shift the whole time |

If your keyboard layout isn't US QWERTY, say which one you use.

## Clean up

```bash
tccutil reset All dev.vocal.spike
rm -rf ~/Applications/VocalSpike.app
```

## What we're finding out

| Question | Steps |
|---|---|
| Does a child process started by the app get the app's permissions? | 2, 3 |
| Started from iTerm, is it attributed to iTerm or to VocalSpike (directly vs. `open`)? | 1, 2 |
| Is spawn, exec or disclaim the right way to start Python? | 3, 4, 5 |
| Do grants survive a rebuild with ad-hoc signing? | 6 |
| Does Accessibility alone cover the keyboard listener? | 7 |
| Can Quartz replace osascript for pasting and typing? | 8 |
