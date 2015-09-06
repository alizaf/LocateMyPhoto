# Locate This View

# Motivations

The idea of estimating image location usign visual features is an inherently rich subject and yet highly unexplored. 
As compared to many other topics in image recognition, geo-recognition requires identification of more key features that remain constant across large spatial scales, making it a challenging and novel task. 

It also represents an integral part of the human experience, where we ourselves possess the innate ability to extract contextual information from the environment in order to draw conclusions about our current location and surroundings. Understanding and achieving visual geo-recognition is therefore crucial towards the development of a more refined and sensitive Artificial Intelligence framework.

In addition of geo-locating images, we would be able to axtract other informatino from scenery images through geo-identification.
Street view images also carry information of propoerties, business locations, news and media that could be directly or indirectly used to add value in various industries. geo-identification can also perform as an laternative to extract geo-tag information from photos, considering the vast majoriy of images do not carry geo-tag information. 

This project is focuse on finding an answer for possibility and accuracy of geo-identifying images. As a result, a Convolutional Neural Network algorithm (LocateThisView) is designed and developed to estimate the exact location of a street view image from San Francisco.

# Method
# Data pipeline

Images for training and testing the algorithm is taken from google street view API. I have collected and calculated lat long information of more than 600 streets ranked based on the number of registered businesses in a given street. Latitude and longitude data for intersecitons are extracted from google geolocation API (using intersection of streets as an address parameter). Finally, regularly distributed points are interpolated every 100 ft, and images with 4 different angles are scraped from google streetview API. 

#Convolutional Neaual Network model

The convolutional neaural network model is designed using three convolutional layers following with two fully connected hidden layers. Nolearn, a python package developed based on Lasagne and Theano, is selected to develop the neaural network model. 

# Results
The neuaral network algorithm of LocateThis View is trained using more than 30000 images and tested on a set of streetview images. I have used these test data to record and analyze the predictions of the model at the end of each epoch. Results are presented as graphs and animations, where we clearly observe how the model learns the features of each area and improves its performance. 
Final model results in more than 70% of the points within 1 km radius from the true value (San Francisco is a 10x10 km area). 

![Alt text][id]
[id]: https://github.com/alizaf/LocateThisView/tree/master/images/200_streets.png "Logo Title Text 2"

<table>
    <tr>
    	<div class="repository-with-sidebar repo-container new-discussion-timeline">
        <div class="repository-sidebar clearfix">
          
<nav class="sunken-menu repo-nav js-repo-nav js-sidenav-container-pjax js-octicon-loaders" role="navigation" data-pjax="#js-repo-pjax-container" data-issue-count-url="/alizaf/LocateThisView/issues/counts">
  <ul class="sunken-menu-group">
    <li class="tooltipped tooltipped-w" aria-label="Code">
      <a href="/alizaf/LocateThisView" aria-label="Code" aria-selected="true" class="js-selected-navigation-item sunken-menu-item selected" data-hotkey="g c" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches /alizaf/LocateThisView">
        <span class="octicon octicon-code"></span> <span class="full-word">Code</span>
        <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16">
</a>    </li>

      <li class="tooltipped tooltipped-w" aria-label="Issues">
        <a href="/alizaf/LocateThisView/issues" aria-label="Issues" class="js-selected-navigation-item sunken-menu-item" data-hotkey="g i" data-selected-links="repo_issues repo_labels repo_milestones /alizaf/LocateThisView/issues">
          <span class="octicon octicon-issue-opened"></span> <span class="full-word">Issues</span>
          <span class="counter">0</span>

          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16">
</a>      </li>

    <li class="tooltipped tooltipped-w" aria-label="Pull requests">
      <a href="/alizaf/LocateThisView/pulls" aria-label="Pull requests" class="js-selected-navigation-item sunken-menu-item" data-hotkey="g p" data-selected-links="repo_pulls /alizaf/LocateThisView/pulls">
          <span class="octicon octicon-git-pull-request"></span> <span class="full-word">Pull requests</span>
          <span class="counter">0</span>

          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16">
</a>    </li>

      <li class="tooltipped tooltipped-w" aria-label="Wiki">
        <a href="/alizaf/LocateThisView/wiki" aria-label="Wiki" class="js-selected-navigation-item sunken-menu-item" data-hotkey="g w" data-selected-links="repo_wiki /alizaf/LocateThisView/wiki">
          <span class="octicon octicon-book"></span> <span class="full-word">Wiki</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16">
</a>      </li>
  </ul>
  <div class="sunken-menu-separator"></div>
  <ul class="sunken-menu-group">

    <li class="tooltipped tooltipped-w" aria-label="Pulse">
      <a href="/alizaf/LocateThisView/pulse" aria-label="Pulse" class="js-selected-navigation-item sunken-menu-item" data-selected-links="pulse /alizaf/LocateThisView/pulse">
        <span class="octicon octicon-pulse"></span> <span class="full-word">Pulse</span>
        <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16">
</a>    </li>

    <li class="tooltipped tooltipped-w" aria-label="Graphs">
      <a href="/alizaf/LocateThisView/graphs" aria-label="Graphs" class="js-selected-navigation-item sunken-menu-item" data-selected-links="repo_graphs repo_contributors /alizaf/LocateThisView/graphs">
        <span class="octicon octicon-graph"></span> <span class="full-word">Graphs</span>
        <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16">
</a>    </li>
  </ul>


    <div class="sunken-menu-separator"></div>
    <ul class="sunken-menu-group">
      <li class="tooltipped tooltipped-w" aria-label="Settings">
        <a href="/alizaf/LocateThisView/settings" aria-label="Settings" class="js-selected-navigation-item sunken-menu-item" data-selected-links="repo_settings repo_branch_settings hooks /alizaf/LocateThisView/settings">
          <span class="octicon octicon-gear"></span> <span class="full-word">Settings</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16">
</a>      </li>
    </ul>
</nav>

            <div class="only-with-full-nav">
                
<div class="js-clone-url clone-url open" data-protocol-type="http">
  <h3><span class="text-emphasized">HTTPS</span> clone URL</h3>
  <div class="input-group js-zeroclipboard-container">
    <input type="text" class="input-mini input-monospace js-url-field js-zeroclipboard-target" value="https://github.com/alizaf/LocateThisView.git" readonly="readonly" aria-label="HTTPS clone URL">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard btn btn-sm zeroclipboard-button tooltipped tooltipped-s" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  
<div class="js-clone-url clone-url " data-protocol-type="ssh">
  <h3><span class="text-emphasized">SSH</span> clone URL</h3>
  <div class="input-group js-zeroclipboard-container">
    <input type="text" class="input-mini input-monospace js-url-field js-zeroclipboard-target" value="git@github.com:alizaf/LocateThisView.git" readonly="readonly" aria-label="SSH clone URL">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard btn btn-sm zeroclipboard-button tooltipped tooltipped-s" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  
<div class="js-clone-url clone-url " data-protocol-type="subversion">
  <h3><span class="text-emphasized">Subversion</span> checkout URL</h3>
  <div class="input-group js-zeroclipboard-container">
    <input type="text" class="input-mini input-monospace js-url-field js-zeroclipboard-target" value="https://github.com/alizaf/LocateThisView" readonly="readonly" aria-label="Subversion checkout URL">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard btn btn-sm zeroclipboard-button tooltipped tooltipped-s" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>



  <div class="clone-options">You can clone with
    <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/users/set_protocol?protocol_selector=http&amp;protocol_type=push" class="inline-form js-clone-selector-form is-enabled" data-form-nonce="ceb78c7a5772a6966452eb4b89041078db810017" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="✓"><input name="authenticity_token" type="hidden" value="dpof5yq4OO/xW9VoiuK8HJUqShI8aJWzfKf8fgptT6cO4eAhhVQNRAZDAqlx+YyohJGE+Wy9fYDsaIf0cLhEnA=="></div><button class="btn-link js-clone-selector" data-protocol="http" type="submit">HTTPS</button></form>, <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/users/set_protocol?protocol_selector=ssh&amp;protocol_type=push" class="inline-form js-clone-selector-form is-enabled" data-form-nonce="ceb78c7a5772a6966452eb4b89041078db810017" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="✓"><input name="authenticity_token" type="hidden" value="Nolmuu45ozxrlgps9r0FWF1bpg1vDkB/yhCqI+uxIeqUioCnXvH7SBmV2tcvE5q4jULxTdU2Zw9wdk5q6hXUWg=="></div><button class="btn-link js-clone-selector" data-protocol="ssh" type="submit">SSH</button></form>, or <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/users/set_protocol?protocol_selector=subversion&amp;protocol_type=push" class="inline-form js-clone-selector-form is-enabled" data-form-nonce="ceb78c7a5772a6966452eb4b89041078db810017" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="✓"><input name="authenticity_token" type="hidden" value="VFVBXA6PxxvpZWKwVdulqnERBpspc57g1xw9i54U6+/EKX9rf8RW0mkLpE1iOKIB9C07zMyNTOKd/vEudRikWw=="></div><button class="btn-link js-clone-selector" data-protocol="subversion" type="submit">Subversion</button></form>.
    <a href="https://help.github.com/articles/which-remote-url-should-i-use" class="help tooltipped tooltipped-n" aria-label="Get help on which URL is right for you.">
      <span class="octicon octicon-question"></span>
    </a>
  </div>
    <a href="github-mac://openRepo/https://github.com/alizaf/LocateThisView" class="btn btn-sm sidebar-button" title="Save alizaf/LocateThisView to your computer and use it in GitHub Desktop." aria-label="Save alizaf/LocateThisView to your computer and use it in GitHub Desktop.">
      <span class="octicon octicon-desktop-download"></span>
      Clone in Desktop
    </a>

              <a href="/alizaf/LocateThisView/archive/master.zip" class="btn btn-sm sidebar-button" aria-label="Download the contents of alizaf/LocateThisView as a zip file" title="Download the contents of alizaf/LocateThisView as a zip file" rel="nofollow">
                <span class="octicon octicon-cloud-download"></span>
                Download ZIP
              </a>
            </div>
        </div>
        <div id="js-repo-pjax-container" class="repository-content context-loader-container" data-pjax-container="">  

<a href="/alizaf/LocateThisView/blob/702ff40f4b8e483bed4a675be24dd2eee38ee6c9/images/200_streets.png" class="hidden js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:b491252453e43c2888f8438e341128b4 -->

  <div class="file-navigation js-zeroclipboard-container">
    
<div class="select-menu js-menu-container js-select-menu left">
  <span class="btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w" data-ref="master" title="master" role="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
    <i>Branch:</i>
    <span class="js-select-button css-truncate-target">master</span>
  </span>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax="" aria-hidden="true">

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <span class="select-menu-title">Switch branches/tags</span>
        <span class="octicon octicon-x js-menu-close" role="button" aria-label="Close"></span>
      </div>

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Find or create a branch…" id="context-commitish-filter-field" class="js-filterable-field js-navigation-enable" placeholder="Find or create a branch…">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" data-filter-placeholder="Find or create a branch…" class="js-select-menu-tab" role="tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" data-filter-placeholder="Find a tag…" class="js-select-menu-tab" role="tab">Tags</a>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches" role="menu">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open selected" href="/alizaf/LocateThisView/blob/master/images/200_streets.png" data-name="master" data-skip-pjax="true" rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="master">
                master
              </span>
            </a>
        </div>

          <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/alizaf/LocateThisView/branches" class="js-create-branch select-menu-item select-menu-new-item-form js-navigation-item js-new-item-form" data-form-nonce="ceb78c7a5772a6966452eb4b89041078db810017" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="✓"><input name="authenticity_token" type="hidden" value="hRp+ua25D8NGuKQl7tqRiwyWJPDruoPAqK5oRx8IfRILjzleHVft0fB2Jxyf3FtasxG1CGJI349hAldo4HNizA=="></div>
            <span class="octicon octicon-git-branch select-menu-item-icon"></span>
            <div class="select-menu-item-text">
              <span class="select-menu-item-heading">Create branch: <span class="js-new-item-name"></span></span>
              <span class="description">from ‘master’</span>
            </div>
            <input type="hidden" name="name" id="name" class="js-new-item-value">
            <input type="hidden" name="branch" id="branch" value="master">
            <input type="hidden" name="path" id="path" value="images/200_streets.png">
</form>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>

    </div>
  </div>
</div>

    <div class="btn-group right">
      <a href="/alizaf/LocateThisView/find/master" class="js-show-file-finder btn btn-sm empty-icon tooltipped tooltipped-nw" data-pjax="" data-hotkey="t" aria-label="Quickly jump between files">
        <span class="octicon octicon-list-unordered"></span>
      </a>
      <button aria-label="Copy file path to clipboard" class="js-zeroclipboard btn btn-sm zeroclipboard-button tooltipped tooltipped-s" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </div>

    <div class="breadcrumb js-zeroclipboard-target">
      <span class="repo-root js-repo-root"><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/alizaf/LocateThisView" class="" data-branch="master" data-pjax="true" itemscope="url"><span itemprop="title">LocateThisView</span></a></span></span><span class="separator">/</span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/alizaf/LocateThisView/tree/master/images" class="" data-branch="master" data-pjax="true" itemscope="url"><span itemprop="title">images</span></a></span><span class="separator">/</span><strong class="final-path">200_streets.png</strong>
    </div>
  </div>


  <div class="commit file-history-tease">
    <div class="file-history-tease-header">
        <img alt="@alizaf" class="avatar" height="24" src="https://avatars1.githubusercontent.com/u/12214059?v=3&amp;s=48" width="24">
        <span class="author"><a href="/alizaf" rel="author">alizaf</a></span>
        <time datetime="2015-08-24T04:03:02Z" is="relative-time" title="August 23, 2015 at 9:03:02 PM PDT">13 days ago</time>
        <div class="commit-title">
            <a href="/alizaf/LocateThisView/commit/ea0fc797dc964cfb01fa2f45d66da56ea13f445b" class="message" data-pjax="true" title="Updates">Updates</a>
        </div>
    </div>

    <div class="participation">
      <p class="quickstat">
        <a href="#blob_contributors_box" rel="facebox">
          <strong>1</strong>
           contributor
        </a>
      </p>
      
    </div>
    <div id="blob_contributors_box" style="display:none">
      <h2 class="facebox-header" data-facebox-id="facebox-header">Users who have contributed to this file</h2>
      <ul class="facebox-user-list" data-facebox-id="facebox-description">
          <li class="facebox-user-list-item">
            <img alt="@alizaf" height="24" src="https://avatars1.githubusercontent.com/u/12214059?v=3&amp;s=48" width="24">
            <a href="/alizaf">alizaf</a>
          </li>
      </ul>
    </div>
  </div>

<div class="file">
  <div class="file-header">
    <div class="file-actions">

      <div class="btn-group">
        <a href="/alizaf/LocateThisView/raw/master/images/200_streets.png" class="btn btn-sm " id="raw-url">Raw</a>
        <a href="/alizaf/LocateThisView/commits/master/images/200_streets.png" class="btn btn-sm " rel="nofollow">History</a>
      </div>

        <a class="octicon-btn tooltipped tooltipped-nw" href="github-mac://openRepo/https://github.com/alizaf/LocateThisView?branch=master&amp;filepath=images%2F200_streets.png" aria-label="Open this file in GitHub Desktop" data-ga-click="Repository, open with desktop, type:mac">
            <span class="octicon octicon-device-desktop"></span>
        </a>


          <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/alizaf/LocateThisView/delete/master/images/200_streets.png" class="inline-form" data-form-nonce="ceb78c7a5772a6966452eb4b89041078db810017" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="✓"><input name="authenticity_token" type="hidden" value="YFMmngyYrs9d8zB8b7DrO456lpKxKzKloSgxMchM7+IjWi4Y9XKHWr5jJh6vaWkLbY4UhV3volWH7dXIOPMltg=="></div>
            <button class="octicon-btn octicon-btn-danger tooltipped tooltipped-n" type="submit" aria-label="Delete this file" data-disable-with="">
              <span class="octicon octicon-trashcan"></span>
            </button>
</form>    </div>

    <div class="file-info">
      207.093 kB
    </div>
  </div>
  

  <div class="blob-wrapper data type-text">
      <div class="image">
          <span class="border-wrap"><img src="/alizaf/LocateThisView/blob/master/images/200_streets.png?raw=true" alt="200_streets.png"></span>
      </div>
  </div>

</div>

<a href="#jump-to-line" rel="facebox[.linejump]" data-hotkey="l" style="display:none">Jump to Line</a>
<div id="jump-to-line" style="display:none">
  <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="" class="js-jump-to-line-form" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="✓"></div>
    <input class="linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line…" aria-label="Jump to line" autofocus="">
    <button type="submit" class="btn">Go</button>
</form></div>




</div>
      </div>
        <td>https://github.com/alizaf/LocateThisView/tree/master/images/200_streets.png</td>
    </tr>
</table>
